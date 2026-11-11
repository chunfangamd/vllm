# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU producer block ownership through the real scheduler/free path."""

import mmap
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum

import msgpack
import msgspec
import pytest
import torch
import zmq
from transformers import LlamaConfig

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio import (
    moriio_common,
    moriio_connector,
    moriio_engine,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOAgentMetadata,
    MoRIIOConnectorMetadata,
    MoRIIOError,
    ReqMeta,
    TransferError,
    get_moriio_read_ack_fenced_hold,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.outputs import KVConnectorOutput

from ..utils import create_model_runner_output, create_request, create_scheduler

pytestmark = pytest.mark.cpu_test
_UNSET = object()


class _LifetimeConfigModel(torch.nn.Module):
    """Registry metadata only: the scheduler never constructs/runs a model."""

    def __init__(self, vllm_config, prefix=""):
        super().__init__()

    def embed_input_ids(self, input_ids):
        return input_ids

    def forward(self, input_ids, positions):
        return input_ids

    def compute_logits(self, hidden_states):
        return hidden_states


def _configuration(tmp_path, *, fenced=True, read=True, role="kv_producer"):
    ModelRegistry.register_model("LifetimeConfigModel", _LifetimeConfigModel)
    LlamaConfig(
        architectures=["LifetimeConfigModel"],
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=1,
        intermediate_size=1024,
        vocab_size=128,
        max_position_embeddings=64,
    ).save_pretrained(tmp_path)
    extras = {
        "read_mode": read,
        "defer_timeout": 1,
        "transfer_timeout": 30,
        "host_ip": "127.0.0.1",
        "backend": "xgmi",
        "http_port": 10001,
        "handshake_port": 10002,
        "notify_port": 10003,
    }
    if fenced is not _UNSET:
        extras["read_ack_fenced_hold"] = fenced
    return VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            max_model_len=64,
            dtype="float32",
            enforce_eager=True,
        ),
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_model_len=64,
            max_num_seqs=4,
            max_num_batched_tokens=64,
            is_encoder_decoder=False,
        ),
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MoRIIOConnector",
            kv_role=role,
            kv_connector_extra_config=extras,
        ),
    )


def _started_producer(tmp_path, monkeypatch, *, fenced=True):
    # Platform/kernel initialization is unrelated to scheduler ownership; this
    # CPU contract test neither compiles kernels nor constructs an engine.
    monkeypatch.setattr(
        type(current_platform), "check_and_update_config", classmethod(lambda *_: None)
    )
    clock = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    core = create_scheduler(_configuration(tmp_path, fenced=fenced), num_blocks=32)
    request = create_request(num_tokens=32, do_remote_decode=True)
    request.kv_transfer_params["transfer_id"] = "t"
    core.add_request(request)
    output = core.schedule()
    return core, request, clock, output


def _finished_producer(tmp_path, monkeypatch, *, fenced=True):
    core, request, clock, output = _started_producer(
        tmp_path, monkeypatch, fenced=fenced
    )
    core.update_from_output(output, create_model_runner_output([request]))
    assert request.is_finished()
    assert request.request_id in core.requests
    assert core.kv_cache_manager.get_blocks(request.request_id).get_block_ids()[0]
    return core, request, clock


def test_fenced_producer_watchdog_cannot_free_pending_reader(tmp_path, monkeypatch):
    core, request, clock = _finished_producer(tmp_path, monkeypatch)
    assert core.connector.connector_scheduler._read_ack_fenced_hold is True
    blocks = core.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
    clock[0] += 2  # producer defer1s expires, consumer transfer30s is still valid
    core._update_from_kv_xfer_finished(KVConnectorOutput())
    assert request.request_id in core.requests
    assert (
        core.kv_cache_manager.get_blocks(request.request_id).get_block_ids() == blocks
    )


@pytest.mark.parametrize("fenced", [False, _UNSET])
def test_legacy_read_producer_still_reaps_on_timeout(tmp_path, monkeypatch, fenced):
    core, request, clock = _finished_producer(tmp_path, monkeypatch, fenced=fenced)
    clock[0] += 2
    core._update_from_kv_xfer_finished(KVConnectorOutput())
    assert request.request_id not in core.requests
    assert core.kv_cache_manager.get_blocks(request.request_id).get_block_ids() == ([],)


def test_missing_ack_never_becomes_quiescence_by_elapsed_time(tmp_path, monkeypatch):
    core, request, clock = _finished_producer(tmp_path, monkeypatch)
    blocks = core.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
    for elapsed in (2, 31, 1000):
        clock[0] += elapsed
        core._update_from_kv_xfer_finished(KVConnectorOutput())
        assert request.request_id in core.requests
        assert (
            core.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
            == blocks
        )


@pytest.mark.parametrize("value", [1, 0, "true", "false", None, [], {}])
def test_fenced_hold_config_rejects_non_bool(value):
    config = KVTransferConfig(
        kv_connector="MoRIIOConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={"read_mode": True, "read_ack_fenced_hold": value},
    )
    with pytest.raises(ValueError, match="must be a bool"):
        get_moriio_read_ack_fenced_hold(config)


def test_old_metadata_cannot_claim_ack_fence():
    fields = dict(
        engine_id="old",
        agent_metadata=b"",
        kv_caches_base_addr=[],
        num_blocks=1,
        block_len=1,
        attn_backend_name="old",
    )
    wire = msgspec.msgpack.encode(fields)
    assert (
        msgspec.msgpack.decode(wire, type=MoRIIOAgentMetadata).read_ack_fenced_hold
        is False
    )
    fields["read_ack_fenced_hold"] = "true"
    with pytest.raises(msgspec.ValidationError):
        msgspec.msgpack.decode(msgspec.msgpack.encode(fields), type=MoRIIOAgentMetadata)


def test_early_terminal_ack_survives_until_request_finished(tmp_path, monkeypatch):
    core, request, clock, output = _started_producer(tmp_path, monkeypatch)
    core._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_sending={request.request_id})
    )
    clock[0] += 2
    core._update_from_kv_xfer_finished(KVConnectorOutput())
    assert request.request_id in core.requests
    assert request.request_id in core.connector.connector_scheduler._pending_sent_acks
    core.update_from_output(output, create_model_runner_output([request]))
    core._update_from_kv_xfer_finished(KVConnectorOutput())
    assert request.request_id not in core.requests


@pytest.mark.parametrize("value", [1, "true"])
def test_real_scheduler_rejects_non_bool_hold(tmp_path, monkeypatch, value):
    monkeypatch.setattr(
        type(current_platform), "check_and_update_config", classmethod(lambda *_: None)
    )
    with pytest.raises(ValueError, match="must be a bool"):
        create_scheduler(_configuration(tmp_path, fenced=value), num_blocks=32)


# The following transport is external to vLLM: real worker/wrapper constructors,
# cache registration, handshake decoding, plan/post/barrier and ACK code all run.
# No native MoRI, GPU, weights, inference, or physical RDMA is exercised.


@dataclass
class _DistributedEndpoint:
    rank: int
    local_rank: int
    cpu_group: None = None


class _Backend(Enum):
    XGMI = 1
    RDMA = 2


class _NativeStatus:
    def __init__(self, result=True):
        self.result = result

    def Succeeded(self):
        return self.result is True

    def Failed(self):
        return self.result is False

    def Message(self):
        return "test native failure" if self.Failed() else ""

    def Code(self):
        return 1 if self.Failed() else 0


@pytest.fixture
def transport(monkeypatch):
    """Fake only the external distributed/MoRI/ZMQ services, never a worker."""

    class Transport:
        rank = 0
        tp = 1

        def __init__(self):
            self.memories = {}
            self.frames = {}
            self.ready = {}
            self.releases = []
            self.posts = []
            self.guards = []
            self.results = []
            self.before_post = None
            self.workers = []
            self.original_wrappers = {}

        def track(self, worker):
            self.workers.append(worker)
            self.original_wrappers[id(worker)] = worker.moriio_wrapper

        @contextmanager
        def socket(self, kind, path):
            port = int(path.rsplit(":", 1)[1])
            bus = self

            class Socket:
                calls = 0

                def send(self, message):
                    if kind == zmq.DEALER:
                        assert bus.ready[port].wait(2)

                def recv_multipart(self):
                    self.calls += 1
                    if kind == zmq.ROUTER:
                        if self.calls == 1:
                            return [b"peer", moriio_common.MoRIIOConstants.GET_META_MSG]
                        raise ExternalTransportClosed
                    return [b"", bus.frames[port][self.calls - 1]]

                def send_multipart(self, values):
                    bus.frames.setdefault(port, []).append(values[-1])
                    if len(bus.frames[port]) == 2:
                        bus.ready.setdefault(port, threading.Event()).set()

            yield Socket()

    bus = Transport()

    class ExternalTransportClosed(BaseException):
        pass

    class ServiceThread(threading.Thread):
        def run(self):
            # This fake endpoint delivered its finite input. Never suppress
            # worker/native errors, only this external service's explicit close.
            with suppress(ExternalTransportClosed):
                super().run()

    monkeypatch.setattr(threading, "Thread", ServiceThread)

    class Memory:
        def __init__(self, tensor):
            self.tensor = tensor.view(torch.uint8).reshape(-1)
            self.data = tensor.data_ptr()
            self.size = self.tensor.numel()
            self.key = str(self.data).encode()
            bus.memories[self.key] = self

        def pack(self):
            return self.key

        @staticmethod
        def unpack(value):
            return bus.memories[value]

    class EngineDesc:
        def __init__(self, key):
            self.key = key

        def pack(self):
            return self.key.encode()

        @classmethod
        def unpack(cls, value):
            return cls(value.decode())

    class Engine:
        def __init__(self, key, config):
            self.key = key
            self.uid = 0

        def create_backend(self, *args):
            pass

        def get_engine_desc(self):
            return EngineDesc(self.key)

        def register_remote_engine(self, desc):
            pass

        def register_torch_tensor(self, tensor):
            return Memory(tensor)

        def allocate_transfer_uid(self):
            self.uid += 1
            return self.uid

        def create_session(self, local, remote):
            class Session:
                def batch_read(self, local_offsets, remote_offsets, sizes, uid):
                    if bus.before_post is not None:
                        bus.before_post(len(bus.posts))
                    bus.posts.append(
                        (local, remote, sizes, local_offsets, remote_offsets)
                    )
                    result = bus.results.pop(0) if bus.results else _NativeStatus()
                    if isinstance(result, Exception):
                        raise result
                    for dst, src, size in zip(
                        local_offsets, remote_offsets, sizes, strict=True
                    ):
                        local.tensor[dst : dst + size] = remote.tensor[src : src + size]
                    return result

            return Session()

    monkeypatch.setattr(
        type(current_platform), "check_and_update_config", classmethod(lambda *_: None)
    )
    for module in (moriio_common, moriio_connector):
        monkeypatch.setattr(
            module,
            "get_tensor_model_parallel_world_size",
            lambda: bus.tp,
            raising=False,
        )
    monkeypatch.setattr(
        moriio_common, "get_tensor_model_parallel_rank", lambda: bus.rank
    )
    monkeypatch.setattr(
        moriio_connector,
        "get_world_group",
        lambda: _DistributedEndpoint(bus.rank, bus.rank),
    )
    monkeypatch.setattr(
        moriio_connector,
        "get_tp_group",
        lambda: _DistributedEndpoint(bus.rank, bus.rank),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(moriio_connector, "MoRIIO_enabled", True)
    monkeypatch.setattr(moriio_connector, "IOEngine", Engine, raising=False)
    monkeypatch.setattr(
        moriio_connector, "IOEngineConfig", lambda *args: args, raising=False
    )
    for module in (moriio_connector, moriio_engine):
        monkeypatch.setattr(module, "BackendType", _Backend, raising=False)
        monkeypatch.setattr(module, "zmq_ctx", bus.socket)
    monkeypatch.setattr(moriio_engine, "XgmiBackendConfig", lambda: None, raising=False)
    monkeypatch.setattr(moriio_engine, "EngineDesc", EngineDesc, raising=False)
    monkeypatch.setattr(moriio_engine, "MemoryDesc", Memory, raising=False)

    # Notifications use a different external socket API. Capture their actual wire.
    class NotifySocket:
        def __init__(self, path):
            self.path = path

        def send(self, value):
            bus.releases.append((self.path, msgpack.loads(value)))

        def close(self, **kwargs):
            pass

    monkeypatch.setattr(
        moriio_engine, "make_zmq_socket", lambda **kwargs: NotifySocket(kwargs["path"])
    )
    yield bus
    for worker in bus.workers:
        worker.shutdown()
        if worker._moriio_handshake_listener_t is not None:
            worker._moriio_handshake_listener_t.join(2)
            assert not worker._moriio_handshake_listener_t.is_alive()
        original_wrapper = bus.original_wrappers[id(worker)]
        original_wrapper.shutdown()
        if original_wrapper.notify_thread is not None:
            original_wrapper.notify_thread.join(2)
            assert not original_wrapper.notify_thread.is_alive()


def _native_worker(
    tmp_path,
    transport,
    *,
    tp=1,
    rank=0,
    producer=False,
    fenced=True,
    legacy=False,
    read=True,
):
    """Construct/configure/register a real worker through its public API."""
    transport.rank, transport.tp = rank, tp
    directory = tmp_path / f"{'producer' if producer else 'consumer'}-{rank}-{tp}"
    directory.mkdir()
    config = _configuration(
        directory,
        fenced=fenced,
        role="kv_producer" if producer else "kv_consumer",
        read=read,
    )
    config.kv_transfer_config.kv_connector_extra_config["transfer_timeout"] = 0.001
    config.kv_transfer_config.kv_connector_extra_config["recv_abort_timeout"] = 0.0001
    config = VllmConfig(
        model_config=config.model_config,
        cache_config=config.cache_config,
        scheduler_config=config.scheduler_config,
        device_config=config.device_config,
        kv_transfer_config=config.kv_transfer_config,
        parallel_config=ParallelConfig(tensor_parallel_size=tp),
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=8 // tp, head_size=128, dtype=torch.float32
    )
    cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer0", "layer1"], spec),
        ],
    )
    worker = moriio_connector.MoRIIOConnectorWorker(
        config, f"{'p' if producer else 'd'}-{rank}", cache_config
    )
    transport.track(worker)
    # mmap supplies page-aligned anonymous CPU memory. Both current canonical
    # byte pages and legacy split K/V views keep a shared-MR middle guard.
    nbytes = spec.page_size_bytes * 4
    buffer = mmap.mmap(-1, 3 * nbytes)
    backing = torch.frombuffer(buffer, dtype=torch.uint8)
    backing.fill_(254)
    if legacy:
        floats = backing.view(torch.float32)
        shape = (2, 4, 16, 8 // tp, 128)
        caches = {
            name: floats[
                index * 2 * nbytes // 4 : (index * 2 + 1) * nbytes // 4
            ].reshape(shape)
            for index, name in enumerate(("layer0", "layer1"))
        }
    else:
        strides = compute_layout_strides(spec, 4, 1, KVCacheLayout.LBNHC)[1:]
        shape = (4, spec.num_heads, spec.num_states, spec.state_content_size_bytes)
        caches = {
            name: torch.as_strided(backing, shape, strides, index * 2 * nbytes)
            for index, name in enumerate(("layer0", "layer1"))
        }
    transport.guards.append(backing[nbytes : 2 * nbytes])
    for index, cache in enumerate(caches.values()):
        if producer:
            values = torch.arange(rank * (8 // tp), (rank + 1) * (8 // tp)) + index * 10
            cache.copy_(
                values.view(1, 1, 1, -1, 1) if legacy else values.view(1, -1, 1, 1)
            )
        else:
            cache.fill_(255)
    worker.register_kv_caches(caches)
    assert transport.ready[worker.side_channel_port].wait(2)
    worker._moriio_handshake_listener_t.join(2)
    return worker, caches


def _native_pair(
    tmp_path,
    transport,
    *,
    producer_tp=2,
    consumer_tp=1,
    consumer_rank=0,
    last_fenced=True,
    legacy=False,
):
    producers = [
        _native_worker(
            tmp_path,
            transport,
            tp=producer_tp,
            rank=rank,
            producer=True,
            fenced=last_fenced if rank == producer_tp - 1 else True,
            legacy=legacy,
        )
        for rank in range(producer_tp)
    ]
    consumer, caches = _native_worker(
        tmp_path, transport, tp=consumer_tp, rank=consumer_rank, legacy=legacy
    )
    meta = MoRIIOConnectorMetadata()
    meta.reqs_to_recv["req"] = ReqMeta(
        transfer_id="t",
        local_block_ids=[[2]],
        remote_block_ids=[[1]],
        remote_host="127.0.0.1",
        remote_port=0,
        remote_handshake_port=10002,
        remote_notify_port=10003,
        remote_engine_id="127.0.0.1:10002",
        tp_size=producer_tp,
        remote_dp_size=1,
    )
    return producers, consumer, caches, meta


@pytest.mark.parametrize(
    ("producer_tp", "consumer_tp", "consumer_rank"),
    [
        (2, 1, 0),
        (1, 2, 0),
        (1, 2, 1),
    ],
)
@pytest.mark.parametrize("legacy", [False, True], ids=["canonical", "legacy-split-kv"])
def test_real_worker_gathers_and_scatters_all_eight_heads(
    tmp_path, transport, producer_tp, consumer_tp, consumer_rank, legacy
):
    _, consumer, caches, metadata = _native_pair(
        tmp_path,
        transport,
        producer_tp=producer_tp,
        consumer_tp=consumer_tp,
        consumer_rank=consumer_rank,
        legacy=legacy,
    )
    consumer.start_load_kv(metadata)
    assert consumer.get_finished() == (set(), set())
    assert len(transport.posts) == 2 * producer_tp
    assert len(transport.releases) == producer_tp
    assert all(torch.all(gap == 254) for gap in transport.guards)
    for index, cache in enumerate(caches.values()):
        heads = (
            torch.arange(
                consumer_rank * (8 // consumer_tp),
                (consumer_rank + 1) * (8 // consumer_tp),
            )
            + index * 10
        )
        received = cache[:, 2] if legacy else cache[2]
        shape = (1, 1, -1, 1) if legacy else (-1, 1, 1)
        expected = heads.view(shape).expand_as(received)
        assert torch.equal(received, expected)
        untouched = cache[:, [0, 1, 3]] if legacy else cache[[0, 1, 3]]
        assert torch.equal(untouched, torch.full_like(untouched, 255))


def test_real_worker_unfenced_later_peer_posts_nothing(tmp_path, transport):
    _, consumer, caches, metadata = _native_pair(tmp_path, transport, last_fenced=False)
    with pytest.raises(MoRIIOError, match="ACK-fenced"):
        consumer.start_load_kv(metadata)
    assert transport.posts == []
    assert transport.releases == []
    assert all(torch.all(cache == 255) for cache in caches.values())


def test_real_worker_pending_survives_new_step_and_expiry(tmp_path, transport):
    _, consumer, _, metadata = _native_pair(tmp_path, transport)
    pending = _NativeStatus(None)
    transport.results = [pending]
    with pytest.raises(TransferError):
        consumer.start_load_kv(metadata)
    assert consumer.get_finished() == (set(), set())
    assert transport.releases == []
    with pytest.raises(TransferError):
        consumer.start_load_kv(MoRIIOConnectorMetadata())
    assert transport.releases == []
    pending.result = True
    consumer.start_load_kv(MoRIIOConnectorMetadata())
    consumer.get_finished()
    assert len(transport.releases) == 2


def test_real_worker_partial_post_keeps_unsealed_request(tmp_path, transport):
    _, consumer, _, metadata = _native_pair(tmp_path, transport)
    transport.results = [_NativeStatus(), RuntimeError("native post state unknown")]
    with pytest.raises(RuntimeError, match="state unknown"):
        consumer.start_load_kv(metadata)
    assert len(transport.posts) == 2
    assert consumer.get_finished() == (set(), set())
    assert transport.releases == []
    with pytest.raises(TransferError, match="unknown native state"):
        consumer.start_load_kv(MoRIIOConnectorMetadata())
    assert transport.releases == []


@pytest.mark.parametrize("defect", ["heads", "layout", "block_count", "session"])
def test_real_worker_bad_later_peer_or_layer_posts_nothing(tmp_path, transport, defect):
    _, consumer, caches, metadata = _native_pair(tmp_path, transport)
    # Alter only the fake external producer's published wire, never local
    # worker fields or the offset/handshake implementation under test.
    frames = transport.frames[10003]  # producer TP rank1
    wire = msgspec.msgpack.decode(frames[0])
    if defect == "heads":
        wire["kv_head_layouts"]["layer1"][0] += 1
    elif defect == "layout":
        wire["kv_head_layouts"]["layer1"][4] = 0
    elif defect == "block_count":
        wire["num_blocks"] = 1
    else:
        layers = msgpack.loads(frames[1])
        layers["layer1"] = []
        frames[1] = msgpack.dumps(layers)
    frames[0] = msgspec.msgpack.encode(wire)
    with pytest.raises((MoRIIOError, ValueError)):
        consumer.start_load_kv(metadata)
    assert transport.posts == []
    assert transport.releases == []
    assert all(torch.all(cache == 255) for cache in caches.values())


def test_real_worker_completed_first_layer_cannot_release_unsealed_request(
    tmp_path, transport
):
    _, consumer, _, metadata = _native_pair(tmp_path, transport)

    def observe_before_later_layer(post_number):
        if post_number == 2:
            assert consumer.get_finished() == (set(), set())
            assert transport.releases == []

    transport.before_post = observe_before_later_layer
    consumer.start_load_kv(metadata)
    assert transport.releases == []
    consumer.get_finished()
    assert len(transport.releases) == 2


def test_real_worker_failed_peer_cannot_release_still_pending_peer(tmp_path, transport):
    _, consumer, _, metadata = _native_pair(tmp_path, transport)
    pending = _NativeStatus(None)
    transport.results = [_NativeStatus(False), pending]
    with pytest.raises(TransferError):
        consumer.start_load_kv(metadata)
    consumer.get_finished()
    assert transport.releases == []
    pending.result = True
    # A terminal attention failure reports load errors but all native writes
    # have stopped, so the actual completion path may now release producers.
    consumer.start_load_kv(MoRIIOConnectorMetadata())
    consumer.get_finished()
    assert len(transport.releases) == 2
    assert consumer.get_block_ids_with_load_errors() == {2}


def test_real_worker_zero_block_abort_cannot_release_existing_pending_read(
    tmp_path, transport
):
    _, consumer, _, metadata = _native_pair(tmp_path, transport)
    transport.results = [_NativeStatus(None)]
    with pytest.raises(TransferError):
        consumer.start_load_kv(metadata)
    metadata.reqs_to_recv["req"].local_block_ids = []
    with pytest.raises(MoRIIOError, match="still owns"):
        consumer.start_load_kv(metadata)
    assert transport.releases == []


@pytest.mark.parametrize("terminal_success", [True, False])
def test_actual_terminal_release_fan_in_frees_real_core_only_after_last_consumer(
    tmp_path, monkeypatch, transport, terminal_success
):
    producers, first, _, metadata = _native_pair(
        tmp_path, transport, producer_tp=1, consumer_tp=2
    )
    second, _ = _native_worker(tmp_path, transport, tp=2, rank=1)
    consumers = [first, second]
    for consumer in consumers:
        transport.results = [
            _NativeStatus(terminal_success),
            _NativeStatus(terminal_success),
        ]
        consumer.start_load_kv(metadata)
    core, request, clock = _finished_producer(tmp_path, monkeypatch)
    blocks = core.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
    free_before = core.kv_cache_manager.block_pool.get_num_free_blocks()
    producer = producers[0][0]
    outgoing = MoRIIOConnectorMetadata()
    outgoing.transfer_id_to_request_id["t"] = request.request_id
    producer.start_load_kv(outgoing)
    clock[0] += 2
    core._update_from_kv_xfer_finished(KVConnectorOutput())
    assert (
        core.kv_cache_manager.get_blocks(request.request_id).get_block_ids() == blocks
    )
    for index, consumer in enumerate(consumers):
        consumer.get_finished()
        _, actual_wire = transport.releases[-1]
        producer.moriio_wrapper._handle_message(msgpack.dumps(actual_wire))
        sending, _ = producer.get_finished()
        assert sending == ({request.request_id} if index == 1 else set())
        core._update_from_kv_xfer_finished(KVConnectorOutput(finished_sending=sending))
        if index == 0:
            assert (
                core.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
                == blocks
            )
    assert request.request_id not in core.requests
    assert core.kv_cache_manager.block_pool.get_num_free_blocks() == free_before + len(
        blocks[0]
    )


@pytest.mark.parametrize(
    ("producer", "fenced", "read", "expected"),
    [
        (True, _UNSET, True, False),
        (True, False, True, False),
        (True, True, True, True),
        (True, True, False, False),
        (False, True, True, False),
    ],
)
def test_real_registration_advertises_only_configured_producer_contract(
    tmp_path, transport, producer, fenced, read, expected
):
    worker, _ = _native_worker(
        tmp_path, transport, producer=producer, fenced=fenced, read=read
    )
    actual_wire = transport.frames[worker.side_channel_port][0]
    metadata = msgspec.msgpack.decode(actual_wire, type=MoRIIOAgentMetadata)
    assert metadata.read_ack_fenced_hold is expected
