"""Read-only discovery and minimal dense FNet injection hooks for Exp5-2."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable


FEATURE_METADATA_KEYS = frozenset(
    ("frame_id", "timestamp", "feature_name", "tensor_shape", "dtype", "device")
)
PRIMARY_FEATURE = "patchifier.fnet.output"
OPTIONAL_FEATURES = (
    "patchifier.forward.fmap",
    "patchifier.forward.gmap",
)


class UnsupportedHookError(RuntimeError):
    """Raised when the required FNet output boundary cannot be observed."""


class PrimaryHookCoverageError(RuntimeError):
    """Raised when the primary hook does not run exactly once for a frame."""


class InjectionContractError(RuntimeError):
    """Raised when a projected latent cannot be safely consumed by FNet."""


@dataclass
class _WrappedForward:
    target: Any
    original: Callable[..., Any]
    had_instance_attribute: bool
    instance_value: Any


def tensor_metadata(tensor: Any, *, frame: dict[str, int], feature_name: str) -> dict[str, Any]:
    """Describe a tensor-like value without reading, cloning, or moving its data."""
    if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype") or not hasattr(tensor, "device"):
        raise TypeError(f"{feature_name} is not tensor-like")
    result = {
        "frame_id": int(frame["frame_id"]),
        "timestamp": int(frame["timestamp"]),
        "feature_name": str(feature_name),
        "tensor_shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }
    if set(result) != FEATURE_METADATA_KEYS:
        raise AssertionError("feature metadata schema changed")
    return result


class FeatureHookRecorder:
    """Observe Patchifier boundaries while returning every upstream value unchanged."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.current_frame: dict[str, int] | None = None
        self.primary_counts: dict[tuple[int, int], int] = {}
        self.registration_modes: dict[str, str] = {}
        self.optional_diagnostics: dict[str, str] = {}
        self.cleanup_errors: list[str] = []
        self._handles: list[Any] = []
        self._wrapped: list[_WrappedForward] = []
        self.initialized = False
        self.removed = False
        self.cleanup_passed = False
        self._installed = False

    @staticmethod
    def _frame_key(frame: dict[str, int]) -> tuple[int, int]:
        return int(frame["frame_id"]), int(frame["timestamp"])

    def _install_observer(
        self, target: Any, callback: Callable[[Any, tuple[Any, ...], Any], None], name: str,
    ) -> None:
        register = getattr(target, "register_forward_hook", None)
        if callable(register):
            handle = register(callback)
            if not hasattr(handle, "remove"):
                raise UnsupportedHookError(f"{name} returned an invalid hook handle")
            self._handles.append(handle)
            self.registration_modes[name] = "register_forward_hook"
            return

        original = getattr(target, "forward", None)
        if not callable(original):
            raise UnsupportedHookError(f"{name} exposes neither forward hook nor callable forward")
        instance_dict = getattr(target, "__dict__", {})
        had_instance_attribute = "forward" in instance_dict
        instance_value = instance_dict.get("forward")

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            output = original(*args, **kwargs)
            callback(target, args, output)
            return output

        try:
            setattr(target, "forward", wrapped)
        except Exception as error:
            raise UnsupportedHookError(f"cannot wrap {name}.forward: {error}") from error
        self._wrapped.append(_WrappedForward(
            target=target,
            original=original,
            had_instance_attribute=had_instance_attribute,
            instance_value=instance_value,
        ))
        self.registration_modes[name] = "forward_wrapper"

    def install(self, patchifier: Any) -> None:
        if self._installed:
            raise RuntimeError("feature hooks are already installed")
        fnet = getattr(patchifier, "fnet", None)
        if fnet is None:
            raise UnsupportedHookError("Patchifier exposes no fnet boundary")
        try:
            self._install_observer(fnet, self._on_fnet, PRIMARY_FEATURE)
            try:
                self._install_observer(
                    patchifier, self._on_patchifier, "patchifier.forward",
                )
            except UnsupportedHookError as error:
                for name in OPTIONAL_FEATURES:
                    self.optional_diagnostics[name] = str(error)
            self._installed = True
            self.initialized = True
            self.removed = False
            self.cleanup_passed = False
        except Exception:
            self.close()
            raise

    def begin_frame(self, record: Any) -> None:
        if not self._installed:
            raise RuntimeError("feature hooks are not installed")
        if self.current_frame is not None:
            raise RuntimeError("nested frame observation is not allowed")
        frame = {
            "frame_id": int(record.frame_id if hasattr(record, "frame_id") else record["frame_id"]),
            "timestamp": int(record.timestamp if hasattr(record, "timestamp") else record["timestamp"]),
        }
        key = self._frame_key(frame)
        if key in self.primary_counts:
            raise RuntimeError(f"duplicate frame observation: {key}")
        self.primary_counts[key] = 0
        self.current_frame = frame

    def record_image(self, image: Any) -> None:
        if self.current_frame is None:
            raise RuntimeError("input image observed outside a frame")
        self.records.append(
            tensor_metadata(image, frame=self.current_frame, feature_name="dpvo.input.image")
        )

    def _on_fnet(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if self.current_frame is None:
            raise RuntimeError("FNet output observed outside a frame")
        self.records.append(
            tensor_metadata(output, frame=self.current_frame, feature_name=PRIMARY_FEATURE)
        )
        key = self._frame_key(self.current_frame)
        self.primary_counts[key] += 1

    def _on_patchifier(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if self.current_frame is None:
            raise RuntimeError("Patchifier output observed outside a frame")
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            detail = f"unsupported Patchifier output: {type(output).__name__}"
            for name in OPTIONAL_FEATURES:
                self.optional_diagnostics[name] = detail
            return
        for name, value in zip(OPTIONAL_FEATURES, output[:2]):
            try:
                self.records.append(
                    tensor_metadata(value, frame=self.current_frame, feature_name=name)
                )
            except (TypeError, ValueError) as error:
                self.optional_diagnostics[name] = str(error)

    def end_frame(self) -> None:
        if self.current_frame is None:
            raise RuntimeError("end_frame called without begin_frame")
        key = self._frame_key(self.current_frame)
        count = self.primary_counts[key]
        self.current_frame = None
        if count != 1:
            raise PrimaryHookCoverageError(
                f"primary FNet hook count for frame {key} is {count}, expected 1"
            )

    def close(self) -> None:
        errors: list[str] = []
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except Exception as error:
                errors.append(repr(error))
        self._handles.clear()
        for wrapped in reversed(self._wrapped):
            try:
                if wrapped.had_instance_attribute:
                    setattr(wrapped.target, "forward", wrapped.instance_value)
                else:
                    delattr(wrapped.target, "forward")
            except Exception as error:
                errors.append(repr(error))
        self._wrapped.clear()
        self._installed = False
        self.current_frame = None
        self.removed = True
        self.cleanup_passed = not errors
        self.cleanup_errors = errors

    def lifecycle(self) -> dict[str, bool]:
        return {
            "initialized": bool(self.initialized),
            "removed": bool(self.removed),
            "cleanup_passed": bool(self.cleanup_passed),
        }

    def __enter__(self) -> "FeatureHookRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class FNetInjectionHook:
    """Add one fmap-scale dense projection to one raw FNet output per frame.

    The hook never stores FNet output. It validates the frame binding and returns
    a tensor with the same shape, dtype and device as the upstream value.
    """

    def __init__(
        self, *, alpha: float = 0.1, channels: int = 128, projection_to_raw_scale: float = 4.0,
    ) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("fusion alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.channels = int(channels)
        self.projection_to_raw_scale = float(projection_to_raw_scale)
        if self.projection_to_raw_scale <= 0.0:
            raise ValueError("projection_to_raw_scale must be positive")
        self.current_frame: tuple[int, int] | None = None
        self._projection: Any | None = None
        self._consumed = 0
        self.frame_count = 0
        self.total_seconds = 0.0
        self.registration_mode: str | None = None
        self.initialized = False
        self.removed = False
        self.cleanup_passed = False
        self.cleanup_errors: list[str] = []
        self._handle: Any | None = None
        self._wrapped: _WrappedForward | None = None
        self._installed = False

    @staticmethod
    def _identity(record: Any) -> tuple[int, int]:
        if hasattr(record, "frame_id"):
            return int(record.frame_id), int(record.timestamp)
        return int(record["frame_id"]), int(record["timestamp"])

    def install(self, fnet: Any) -> None:
        if self._installed:
            raise RuntimeError("FNet injection hook is already installed")
        register = getattr(fnet, "register_forward_hook", None)
        try:
            if callable(register):
                try:
                    handle = register(self._on_fnet, always_call=False)
                except TypeError:
                    handle = register(self._on_fnet)
                if not hasattr(handle, "remove"):
                    raise UnsupportedHookError("FNet returned an invalid injection hook handle")
                self._handle = handle
                self.registration_mode = "register_forward_hook"
            else:
                original = getattr(fnet, "forward", None)
                if not callable(original):
                    raise UnsupportedHookError(
                        "FNet exposes neither register_forward_hook nor callable forward"
                    )
                instance_dict = getattr(fnet, "__dict__", {})
                state = _WrappedForward(
                    target=fnet,
                    original=original,
                    had_instance_attribute="forward" in instance_dict,
                    instance_value=instance_dict.get("forward"),
                )

                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    return self._fuse(original(*args, **kwargs))

                setattr(fnet, "forward", wrapped)
                self._wrapped = state
                self.registration_mode = "forward_wrapper"
            self._installed = True
            self.initialized = True
            self.removed = False
            self.cleanup_passed = False
        except Exception:
            self.close()
            raise

    def bind(self, record: Any, projected_jepa: Any) -> None:
        if not self._installed:
            raise RuntimeError("FNet injection hook is not installed")
        if self.current_frame is not None:
            raise RuntimeError("previous projected JEPA value was not finalized")
        import torch

        if not isinstance(projected_jepa, torch.Tensor):
            raise InjectionContractError("projected JEPA value must be a torch.Tensor")
        if projected_jepa.ndim != 4 or int(projected_jepa.shape[1]) != self.channels:
            raise InjectionContractError(
                f"expected projected dense JEPA [B,{self.channels},H,W], "
                f"got {tuple(projected_jepa.shape)}"
            )
        if not bool(torch.isfinite(projected_jepa).all().item()):
            raise InjectionContractError("projected JEPA value contains non-finite entries")
        self.current_frame = self._identity(record)
        self._projection = projected_jepa
        self._consumed = 0

    def _on_fnet(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return self._fuse(output)

    def _fuse(self, output: Any) -> Any:
        import torch

        if self.current_frame is None or self._projection is None:
            raise InjectionContractError("FNet executed without a bound frame projection")
        if self._consumed:
            raise InjectionContractError(
                f"projection for frame {self.current_frame} was consumed more than once"
            )
        if not isinstance(output, torch.Tensor) or output.ndim != 5:
            raise InjectionContractError("FNet output must be a [B,N,C,H,W] torch.Tensor")
        batch, frames, channels = (int(output.shape[index]) for index in range(3))
        if channels != self.channels:
            raise InjectionContractError(
                f"FNet channel mismatch: {channels} != {self.channels}"
            )
        projection = self._projection
        if tuple(projection.shape[-2:]) != tuple(output.shape[-2:]):
            raise InjectionContractError(
                f"dense projection spatial mismatch: {tuple(projection.shape[-2:])} "
                f"!= {tuple(output.shape[-2:])}"
            )
        if int(projection.shape[0]) == batch and frames == 1:
            projection = projection[:, None]
        elif int(projection.shape[0]) == batch * frames:
            projection = projection.reshape(
                batch, frames, channels, int(output.shape[-2]), int(output.shape[-1])
            )
        else:
            raise InjectionContractError(
                f"projection batch {projection.shape[0]} cannot bind FNet [B,N]=[{batch},{frames}]"
            )
        started = time.perf_counter()
        residual = projection.to(device=output.device, dtype=output.dtype)
        fused = output + self.alpha * self.projection_to_raw_scale * residual
        if tuple(fused.shape) != tuple(output.shape) or fused.dtype != output.dtype:
            raise InjectionContractError("fusion changed FNet shape or dtype")
        if not bool(torch.isfinite(fused).all().item()):
            raise InjectionContractError("fused FNet output contains non-finite entries")
        if output.device.type == "cuda":
            torch.cuda.synchronize(output.device)
        self.total_seconds += time.perf_counter() - started
        self._consumed += 1
        return fused

    def finish_frame(self, record: Any) -> None:
        identity = self._identity(record)
        if self.current_frame != identity:
            raise InjectionContractError(
                f"frame identity mismatch: bound={self.current_frame}, finished={identity}"
            )
        if self._consumed != 1:
            raise PrimaryHookCoverageError(
                f"FNet injection count for frame {identity} is {self._consumed}, expected 1"
            )
        self.frame_count += 1
        self.current_frame = None
        self._projection = None
        self._consumed = 0

    def close(self) -> None:
        errors: list[str] = []
        if self.current_frame is not None:
            errors.append(f"unconsumed frame binding: {self.current_frame}")
        if self._handle is not None:
            try:
                self._handle.remove()
            except Exception as error:
                errors.append(repr(error))
            self._handle = None
        if self._wrapped is not None:
            try:
                if self._wrapped.had_instance_attribute:
                    setattr(self._wrapped.target, "forward", self._wrapped.instance_value)
                else:
                    delattr(self._wrapped.target, "forward")
            except Exception as error:
                errors.append(repr(error))
            self._wrapped = None
        self._installed = False
        self.current_frame = None
        self._projection = None
        self._consumed = 0
        self.removed = True
        self.cleanup_errors = errors
        self.cleanup_passed = not errors

    def lifecycle(self) -> dict[str, Any]:
        return {
            "initialized": bool(self.initialized),
            "removed": bool(self.removed),
            "cleanup_passed": bool(self.cleanup_passed),
            "registration_mode": self.registration_mode,
            "consumed_frames": int(self.frame_count),
        }

    def __enter__(self) -> "FNetInjectionHook":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class OracleFNetReplacementHook:
    """Replace non-keyframe raw FNet output with scale-equivalent dense JEPA.

    Keyframes preserve the Exp5-2 residual fusion contract. Non-keyframes do
    not call the wrapped FNet implementation: the wrapper returns a projected
    tensor with the same ``[B,N,C,H,W]`` interface expected by Patchifier.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        channels: int = 128,
        projection_to_raw_scale: float = 4.0,
        replacement_dtype: Any | None = None,
    ) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("oracle keyframe alpha must be in (0, 1]")
        if float(projection_to_raw_scale) <= 0.0:
            raise ValueError("oracle projection scale must be positive")
        self.alpha = float(alpha)
        self.channels = int(channels)
        self.projection_to_raw_scale = float(projection_to_raw_scale)
        self.replacement_dtype = replacement_dtype
        self.current_frame: tuple[int, int] | None = None
        self.current_is_keyframe: bool | None = None
        self._projection: Any | None = None
        self._consumed = 0
        self.frame_count = 0
        self.fnet_executed_frames = 0
        self.jepa_replaced_frames = 0
        self.total_seconds = 0.0
        self.registration_mode = "forward_wrapper"
        self.initialized = False
        self.removed = False
        self.cleanup_passed = False
        self.cleanup_errors: list[str] = []
        self._wrapped: _WrappedForward | None = None
        self._installed = False
        self._output_contract: tuple[int, ...] | None = None

    @staticmethod
    def _identity(record: Any) -> tuple[int, int]:
        if hasattr(record, "frame_id"):
            return int(record.frame_id), int(record.timestamp)
        return int(record["frame_id"]), int(record["timestamp"])

    def install(self, fnet: Any) -> None:
        if self._installed:
            raise RuntimeError("Oracle FNet replacement is already installed")
        original = getattr(fnet, "forward", None)
        if not callable(original):
            raise UnsupportedHookError("FNet exposes no callable forward for Oracle replacement")
        instance_dict = getattr(fnet, "__dict__", {})
        state = _WrappedForward(
            target=fnet,
            original=original,
            had_instance_attribute="forward" in instance_dict,
            instance_value=instance_dict.get("forward"),
        )

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.current_frame is None or self.current_is_keyframe is None:
                raise InjectionContractError("FNet reached without an Oracle frame binding")
            if self.current_is_keyframe:
                return self._consume(original(*args, **kwargs), replacement=False)
            return self._consume(None, replacement=True)

        try:
            setattr(fnet, "forward", wrapped)
            self._wrapped = state
            self._installed = True
            self.initialized = True
            self.removed = False
            self.cleanup_passed = False
        except Exception:
            self.close()
            raise

    def bind(self, record: Any, projected_jepa: Any, *, is_keyframe: bool) -> None:
        if not self._installed:
            raise RuntimeError("Oracle FNet replacement is not installed")
        if self.current_frame is not None:
            raise RuntimeError("previous Oracle projection was not finalized")
        import torch

        if not isinstance(projected_jepa, torch.Tensor):
            raise InjectionContractError("Oracle projected JEPA must be a torch.Tensor")
        if projected_jepa.ndim != 4 or int(projected_jepa.shape[1]) != self.channels:
            raise InjectionContractError(
                f"expected Oracle projection [B,{self.channels},H,W], "
                f"got {tuple(projected_jepa.shape)}"
            )
        if not bool(torch.isfinite(projected_jepa).all().item()):
            raise InjectionContractError("Oracle projection contains non-finite entries")
        self.current_frame = self._identity(record)
        self.current_is_keyframe = bool(is_keyframe)
        self._projection = projected_jepa
        self._consumed = 0

    def _consume(self, original_output: Any | None, *, replacement: bool) -> Any:
        import torch

        if self.current_frame is None or self._projection is None:
            raise InjectionContractError("Oracle FNet executed without a bound projection")
        if self._consumed:
            raise InjectionContractError(
                f"Oracle projection for frame {self.current_frame} was consumed more than once"
            )
        projection = self._projection
        started = time.perf_counter()
        if replacement:
            dtype = self.replacement_dtype or projection.dtype
            output = (
                self.projection_to_raw_scale
                * projection.to(device=projection.device, dtype=dtype)
            )[:, None]
            self.jepa_replaced_frames += 1
        else:
            if not isinstance(original_output, torch.Tensor) or original_output.ndim != 5:
                raise InjectionContractError(
                    "Oracle keyframe FNet output must be [B,N,C,H,W]"
                )
            batch, frames, channels = (
                int(original_output.shape[index]) for index in range(3)
            )
            if channels != self.channels or frames != 1:
                raise InjectionContractError(
                    f"Oracle keyframe FNet contract mismatch: {tuple(original_output.shape)}"
                )
            if tuple(projection.shape) != (
                batch, channels, int(original_output.shape[-2]), int(original_output.shape[-1])
            ):
                raise InjectionContractError("Oracle keyframe projection shape mismatch")
            residual = projection.to(
                device=original_output.device, dtype=original_output.dtype
            )[:, None]
            output = (
                original_output
                + self.alpha * self.projection_to_raw_scale * residual
            )
            self.fnet_executed_frames += 1
        if output.ndim != 5 or int(output.shape[2]) != self.channels:
            raise InjectionContractError("Oracle replacement changed FNet tensor interface")
        contract = tuple(int(value) for value in output.shape)
        if self._output_contract is None:
            self._output_contract = contract
        elif contract != self._output_contract:
            raise InjectionContractError(
                f"Oracle FNet output contract changed: {contract} != {self._output_contract}"
            )
        if not bool(torch.isfinite(output).all().item()):
            raise InjectionContractError("Oracle FNet output contains non-finite entries")
        if output.device.type == "cuda":
            torch.cuda.synchronize(output.device)
        self.total_seconds += time.perf_counter() - started
        self._consumed += 1
        return output

    def finish_frame(self, record: Any, *, is_keyframe: bool) -> None:
        identity = self._identity(record)
        if self.current_frame != identity or self.current_is_keyframe != bool(is_keyframe):
            raise InjectionContractError("Oracle frame identity/keyframe mismatch")
        if self._consumed != 1:
            raise PrimaryHookCoverageError(
                f"Oracle FNet boundary count for frame {identity} is {self._consumed}, expected 1"
            )
        self.frame_count += 1
        self.current_frame = None
        self.current_is_keyframe = None
        self._projection = None
        self._consumed = 0

    def close(self) -> None:
        errors: list[str] = []
        if self.current_frame is not None:
            errors.append(f"unconsumed Oracle frame binding: {self.current_frame}")
        if self._wrapped is not None:
            try:
                if self._wrapped.had_instance_attribute:
                    setattr(self._wrapped.target, "forward", self._wrapped.instance_value)
                else:
                    delattr(self._wrapped.target, "forward")
            except Exception as error:
                errors.append(repr(error))
            self._wrapped = None
        self._installed = False
        self.current_frame = None
        self.current_is_keyframe = None
        self._projection = None
        self._consumed = 0
        self.removed = True
        self.cleanup_errors = errors
        self.cleanup_passed = not errors

    def lifecycle(self) -> dict[str, Any]:
        return {
            "initialized": bool(self.initialized),
            "removed": bool(self.removed),
            "cleanup_passed": bool(self.cleanup_passed),
            "registration_mode": self.registration_mode,
            "consumed_frames": int(self.frame_count),
            "fnet_executed_frames": int(self.fnet_executed_frames),
            "jepa_replaced_frames": int(self.jepa_replaced_frames),
            "output_contract": list(self._output_contract) if self._output_contract else None,
        }

    def __enter__(self) -> "OracleFNetReplacementHook":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
