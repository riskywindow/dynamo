# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Unix FD transfers retain payload/socket ownership without duplicate leaks."""
import asyncio,os,socket,unittest
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import pytest
from gpu_memory_service.common.protocol import wire
from gpu_memory_service.common.protocol.messages import ErrorResponse, ExportAllocationResponse
pytestmark=[pytest.mark.pre_merge,pytest.mark.unit,pytest.mark.gpu_0,pytest.mark.none]

class Writer:
    def __init__(self,sock):self.sock=sock
    def get_extra_info(self,name):return self.sock if name=='socket' else None
    def write(self,data):self.sock.sendall(data)
    async def drain(self):pass

@unittest.skipUnless(os.path.isdir('/proc/self/fd'),'requires Linux descriptor accounting')
class FDTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server,self.client=socket.socketpair();self.server.setblocking(False);self.client.settimeout(3)
        self.writer=Writer(self.server);self.payload=os.open(os.devnull,os.O_RDONLY)
        self.target=os.readlink('/proc/self/fd/'+str(self.server.fileno()))
        await asyncio.get_running_loop().run_in_executor(None,lambda:None)
    async def asyncTearDown(self):
        # Clean test-owned duplicates even when reproducing the old implementation.
        for name in os.listdir('/proc/self/fd'):
            fd=int(name)
            if fd==self.server.fileno():continue
            try:
                if os.readlink('/proc/self/fd/'+name)==self.target:os.close(fd)
            except FileNotFoundError:pass
        self.server.close();self.client.close();os.close(self.payload)
    def count(self):return len(os.listdir('/proc/self/fd'))
    async def test_successful_exports_close_socket_duplicates(self):
        before=self.count()
        for _ in range(100):
            await wire.send_message(self.writer,ErrorResponse(error='fd-lifetime'),self.payload)
            msg,fd,_=wire.recv_message_sync(self.client)
            self.assertEqual(msg.error,'fd-lifetime');self.assertGreaterEqual(fd,0);os.fstat(fd);os.close(fd);os.fstat(self.payload)
        self.assertEqual(self.count(),before)
        # The original transport remains usable and the caller retains payload ownership.
        await wire.send_message(self.writer,ErrorResponse(error='still-open'))
        msg,fd,_=wire.recv_message_sync(self.client);self.assertEqual(msg.error,'still-open');self.assertEqual(fd,-1)
    async def test_send_failure_closes_socket_duplicate(self):
        before=self.count()
        with patch.object(wire.socket.socket,'sendmsg',side_effect=BrokenPipeError('injected')):
            with self.assertRaises(BrokenPipeError):await wire.send_message(self.writer,ErrorResponse(error='test'),self.payload)
        self.assertEqual(self.count(),before);os.fstat(self.payload)
    async def test_socket_construction_failure_closes_duplicate(self):
        before=self.count()
        with patch.object(wire.socket,'socket',side_effect=OSError('injected')):
            with self.assertRaises(OSError):await wire.send_message(self.writer,ErrorResponse(error='test'),self.payload)
        self.assertEqual(self.count(),before);os.fstat(self.payload)


@pytest.mark.timeout(10)
@unittest.skipUnless(
    os.path.isdir('/proc/self/fd') and hasattr(socket, 'SCM_RIGHTS'),
    'requires Linux Unix-socket descriptor passing',
)
class StreamFDTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.loop = asyncio.get_running_loop()
        self.loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        self.server, self.client = socket.socketpair()
        self.addCleanup(self.server.close)
        self.addCleanup(self.client.close)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        self.server.setblocking(False)
        self.client.settimeout(2)
        _, self.writer = await asyncio.open_connection(sock=self.server)
        self.addAsyncCleanup(self.close_transport)
        self.payload = tempfile.TemporaryFile()
        self.addCleanup(self.payload.close)
        self.payload.write(b'ORIGINAL')
        self.payload.flush()
        await self.loop.run_in_executor(None, lambda: None)

    async def close_transport(self):
        self.client.close()
        self.writer.close()
        try:
            await asyncio.wait_for(self.writer.wait_closed(), 2)
        except (BrokenPipeError, ConnectionResetError):
            pass
        # A single-worker queue also joins the control's cancelled executor send.
        await asyncio.wait_for(self.loop.run_in_executor(None, lambda: None), 2)

    async def test_fd_send_preserves_asyncio_socket_nonblocking(self):
        await wire.send_message(
            self.writer, ErrorResponse(error='stream-writer'), self.payload.fileno()
        )
        message, received_fd, remaining = wire.recv_message_sync(self.client)
        try:
            self.assertEqual(message.error, 'stream-writer')
            self.assertFalse(remaining)
            self.assertEqual(os.pread(received_fd, 8, 0), b'ORIGINAL')
            os.fstat(self.payload.fileno())
            self.assertFalse(os.get_blocking(self.server.fileno()))
        finally:
            os.close(received_fd)

    async def test_cancel_blocked_send_reclaims_socket_before_payload_owner(self):
        # Force the real candidate to await writable readiness, without sleeps.
        self.fill_send_buffer()
        entered = asyncio.Event()
        release_control = threading.Event()
        owner_closed = threading.Event()
        loop_thread = threading.get_ident()
        real_sendmsg = socket.socket.sendmsg
        observation = {'used_fd_after_owner_close': False}
        owned_fd = os.dup(self.payload.fileno())

        def traced_sendmsg(sock, buffers, ancillary, *args, **kwargs):
            observation['socket_duplicate'] = sock.fileno()
            self.loop.call_soon_threadsafe(entered.set)
            # Keep the old executor control before its real syscall until the
            # coroutine has cancelled and its caller has closed the payload FD.
            if threading.get_ident() != loop_thread:
                if not release_control.wait(2):
                    raise TimeoutError('control barrier timed out')
            observation['used_fd_after_owner_close'] |= owner_closed.is_set()
            return real_sendmsg(sock, buffers, ancillary, *args, **kwargs)

        async def caller():
            try:
                await wire.send_message(
                    self.writer, ErrorResponse(error='cancelled'), owned_fd
                )
            finally:
                # Matches server/rpc.py's ownership around await send_message.
                os.close(owned_fd)
                owner_closed.set()

        task = asyncio.create_task(caller())
        try:
            with patch.object(socket.socket, 'sendmsg', traced_sendmsg):
                try:
                    await asyncio.wait_for(entered.wait(), 2)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 2)
                    duplicate_alive_at_return = os.path.exists(
                        f'/proc/self/fd/{observation["socket_duplicate"]}'
                    )
                finally:
                    # Release and join the expected failing control before any
                    # assertions, so baseline failures leave no blocked worker.
                    release_control.set()
                    self.client.close()
                    await asyncio.wait_for(
                        self.loop.run_in_executor(None, lambda: None), 2
                    )
        finally:
            if not task.done():
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
        self.assertTrue(owner_closed.is_set())
        os.fstat(self.payload.fileno())
        self.assertFalse(duplicate_alive_at_return)
        self.assertFalse(observation['used_fd_after_owner_close'])

    async def test_partial_export_frame_delivers_exact_message_and_one_fd(self):
        tag = 'large-export:' + 'z' * (256 * 1024)
        message = ExportAllocationResponse(
            allocation_id='stream-test', size=8, aligned_size=8, tag=tag, layout_slot=0
        )
        first_entered = asyncio.Event()
        drain_allowed = threading.Event()
        receiver_done = asyncio.Event()
        real_sendmsg = socket.socket.sendmsg
        observation = {'fd_sends': 0, 'receiver_error': None}

        def traced_sendmsg(sock, buffers, ancillary, *args, **kwargs):
            observation['fd_sends'] += 1
            self.loop.call_soon_threadsafe(first_entered.set)
            count = real_sendmsg(sock, buffers, ancillary, *args, **kwargs)
            observation['first_send_bytes'] = count
            return count

        def receive():
            received_fd = -1
            try:
                if not drain_allowed.wait(2):
                    raise TimeoutError('receiver barrier timed out')
                response, received_fd, remaining = wire.recv_message_sync(self.client)
                observation['received_tag'] = response.tag
                observation['received_payload'] = os.pread(received_fd, 8, 0)
                observation['remaining'] = remaining
            except (OSError, ValueError) as exc:
                observation['receiver_error'] = exc
            finally:
                if received_fd >= 0:
                    os.close(received_fd)
                self.loop.call_soon_threadsafe(receiver_done.set)

        receiver = threading.Thread(target=receive)
        receiver.start()
        task = asyncio.create_task(
            wire.send_message(self.writer, message, self.payload.fileno())
        )
        try:
            with patch.object(socket.socket, 'sendmsg', traced_sendmsg):
                await asyncio.wait_for(first_entered.wait(), 2)
                # The small kernel buffer returns a real short send while the
                # receiver is still held at its barrier.
                first_send_bytes = observation.get('first_send_bytes', 0)
                drain_allowed.set()
                await asyncio.wait_for(task, 3)
            await asyncio.wait_for(receiver_done.wait(), 3)
        finally:
            drain_allowed.set()
            if not task.done():
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
            self.client.close()
            receiver.join(2)
        self.assertFalse(receiver.is_alive())
        self.assertIsNone(observation['receiver_error'])
        self.assertGreater(first_send_bytes, 0)
        self.assertLess(first_send_bytes, len(wire._frame_message(message)))
        self.assertEqual(observation['received_tag'], tag)
        self.assertEqual(observation['received_payload'], b'ORIGINAL')
        self.assertFalse(observation['remaining'])
        self.assertEqual(observation['fd_sends'], 1)
        os.fstat(self.payload.fileno())

    def fill_send_buffer(self):
        buffered = 0
        while True:
            try:
                buffered += self.server.send(b'x' * 4096, socket.MSG_DONTWAIT)
                self.assertLess(buffered, 1024 * 1024)
            except BlockingIOError:
                return

    async def test_cancelled_waiter_allows_reused_fd_to_receive_writable_callback(self):
        self.fill_send_buffer()
        entered = asyncio.Event()
        real_sendmsg = socket.socket.sendmsg
        observation = {}

        def traced_sendmsg(sock, buffers, ancillary, *args, **kwargs):
            observation['socket_duplicate'] = sock.fileno()
            self.loop.call_soon_threadsafe(entered.set)
            return real_sendmsg(sock, buffers, ancillary, *args, **kwargs)

        task = asyncio.create_task(
            wire.send_message(self.writer, ErrorResponse(error='waiter'), self.payload.fileno())
        )
        replacement = peer = None
        reused_fd = None
        try:
            with patch.object(socket.socket, 'sendmsg', traced_sendmsg):
                await asyncio.wait_for(entered.wait(), 2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
            old_fd = observation['socket_duplicate']
            self.assertFalse(os.path.exists(f'/proc/self/fd/{old_fd}'))
            self.assertFalse(self.writer.is_closing())
            replacement, peer = socket.socketpair()
            if peer.fileno() == old_fd:
                replacement, peer = peer, replacement
            if replacement.fileno() == old_fd:
                reused_fd = old_fd
            else:
                reused_fd = os.dup2(replacement.fileno(), old_fd)
            writable = asyncio.Event()
            self.loop.add_writer(reused_fd, writable.set)
            # A stale selector entry for the old socket can prevent registering
            # this writable, unrelated socket at the same descriptor number.
            await asyncio.wait_for(writable.wait(), 1)
        finally:
            if reused_fd is not None:
                self.loop.remove_writer(reused_fd)
                if reused_fd != replacement.fileno():
                    os.close(reused_fd)
            elif 'socket_duplicate' in observation:
                # Also clear a stale entry on an earlier expected failure.
                self.loop.remove_writer(observation['socket_duplicate'])
            if replacement is not None:
                replacement.close()
                peer.close()
            if not task.done():
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
            self.client.close()
            await asyncio.wait_for(self.loop.run_in_executor(None, lambda: None), 2)

    async def test_partial_frame_cancel_closes_peer_and_releases_received_fd(self):
        message = ExportAllocationResponse(
            allocation_id='partial-cancel', size=8, aligned_size=8,
            tag='z' * (256 * 1024), layout_slot=0,
        )
        entered = asyncio.Event()
        real_sendmsg = socket.socket.sendmsg
        observation = {}
        target = os.readlink(f'/proc/self/fd/{self.payload.fileno()}')

        def payload_descriptors():
            descriptors = []
            for name in os.listdir('/proc/self/fd'):
                try:
                    if os.readlink('/proc/self/fd/' + name) == target:
                        descriptors.append(int(name))
                except FileNotFoundError:
                    pass
            return sorted(descriptors)

        def traced_sendmsg(sock, buffers, ancillary, *args, **kwargs):
            self.loop.call_soon_threadsafe(entered.set)
            count = real_sendmsg(sock, buffers, ancillary, *args, **kwargs)
            observation['first_send_bytes'] = count
            return count

        def receive():
            received_fd = -1
            try:
                _, received_fd, _ = wire.recv_message_sync(self.client)
                return 'unexpected_complete_message'
            except ConnectionResetError:
                return 'ConnectionResetError'
            except TimeoutError:
                return 'peer_did_not_close'
            finally:
                if received_fd >= 0:
                    os.close(received_fd)

        before = payload_descriptors()
        self.client.settimeout(1)
        task = asyncio.create_task(
            wire.send_message(self.writer, message, self.payload.fileno())
        )
        receiver = None
        try:
            with patch.object(socket.socket, 'sendmsg', traced_sendmsg):
                await asyncio.wait_for(entered.wait(), 2)
                sent = observation.get('first_send_bytes', 0)
                self.assertGreater(sent, 0)
                self.assertLess(sent, len(wire._frame_message(message)))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
            closing_at_cancel_return = self.writer.is_closing()
            receiver = self.loop.run_in_executor(None, receive)
            # Socket timeout bounds the negative mutant; shielding keeps its
            # receive owner joinable if an outer wait unexpectedly expires.
            outcome = await asyncio.wait_for(asyncio.shield(receiver), 2)
            self.assertTrue(closing_at_cancel_return)
            self.assertEqual(outcome, 'ConnectionResetError')
            self.assertEqual(payload_descriptors(), before)
        finally:
            self.writer.transport.abort()
            self.client.close()
            if not task.done():
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
            if receiver is not None:
                await asyncio.wait_for(asyncio.shield(receiver), 2)
            await asyncio.wait_for(self.loop.run_in_executor(None, lambda: None), 2)
