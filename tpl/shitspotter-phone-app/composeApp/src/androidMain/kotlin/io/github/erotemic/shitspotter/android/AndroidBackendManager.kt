package io.github.erotemic.shitspotter.android

import android.content.Context
import io.github.erotemic.shitspotter.core.AppLogger
import io.github.erotemic.shitspotter.core.DetectorBackend
import io.github.erotemic.shitspotter.core.FrameSource
import io.github.erotemic.shitspotter.core.InferenceResult
import io.github.erotemic.shitspotter.core.ModelFormat
import io.github.erotemic.shitspotter.core.ModelRegistry
import io.github.erotemic.shitspotter.core.ModelSpec
import io.github.erotemic.shitspotter.core.StubDetectorBackend
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Owns the current Android [DetectorBackend] and serialises every
 * frame's analyse call with respect to model swaps. The previous
 * design exposed `current: DetectorBackend` through a volatile getter,
 * but that let the camera thread capture an old reference, then a
 * concurrent `setActive` could close the old backend before
 * `analyze()` ever ran on it.
 *
 * Fixed by keeping the backend reference **inside** the manager:
 *   - [analyze] takes the lock for the full duration of inference, so
 *     a parallel [setActive] blocks until the in-flight frame finishes.
 *   - [setActive] also holds the lock while it closes the old backend
 *     and brings up the new one, so [analyze] can never see a half-
 *     swapped state.
 *
 * The lock granularity is "one analyse OR one swap at a time." For the
 * Pixel-5-class hot path with ~20–60 ms inference and infrequent model
 * swaps that contention is fine. If contention ever becomes
 * measurable, swap this for a ref-counted lease.
 */
class AndroidBackendManager(
    private val context: Context,
    private val logger: AppLogger = AndroidLogger,
) {
    private val lock = ReentrantLock()
    private val loader = AndroidModelLoader(context)
    private var current: DetectorBackend = StubDetectorBackend()
    private var currentId: String = current.spec.modelId

    /** Run inference under the swap-protecting lock. Returns the
     *  per-frame [InferenceResult] **and** a snapshot of the spec that
     *  was active at the time. The caller (CameraAnalysisLoop) uses
     *  the snapshot for telemetry so the HUD never reports a
     *  spec/model-id that disagrees with the actual inference. */
    fun analyze(frame: FrameSource): AnalysisOutput = lock.withLock {
        val b = current
        AnalysisOutput(result = b.analyze(frame), spec = b.spec)
    }

    /** Read-only snapshot of the active backend's display metadata.
     *  Used by `saveFailureCase` and the HUD's first-frame state. */
    fun snapshot(): BackendSnapshot = lock.withLock {
        val b = current
        BackendSnapshot(
            spec = b.spec,
            backendName = b.backendName,
            delegate = b.delegate,
        )
    }

    val activeModelId: String get() = lock.withLock { currentId }

    /**
     * Switch to the model registered as [modelId]. Returns the new
     * backend's snapshot (which may be a stub if the id is unknown or
     * the model file isn't available). Idempotent — calling with the
     * already-active id is a no-op that returns the current snapshot.
     */
    fun setActive(modelId: String): BackendSnapshot = lock.withLock {
        if (modelId == currentId) {
            return@withLock BackendSnapshot(
                spec = current.spec,
                backendName = current.backendName,
                delegate = current.delegate,
            )
        }
        val spec = ModelRegistry.byId(modelId)
        val newBackend: DetectorBackend = when {
            spec == null -> {
                logger.warn(TAG, "unknown modelId '$modelId'; staying on $currentId")
                return@withLock BackendSnapshot(
                    spec = current.spec,
                    backendName = current.backendName,
                    delegate = current.delegate,
                )
            }
            spec.format == ModelFormat.STUB -> StubDetectorBackend(spec)
            spec.format == ModelFormat.ONNX -> tryLoadOnnx(spec) ?: StubDetectorBackend(spec)
            else -> {
                logger.warn(
                    TAG,
                    "unsupported format ${spec.format} for modelId '$modelId'; falling back to stub",
                )
                StubDetectorBackend(spec)
            }
        }
        val old = current
        current = newBackend
        currentId = modelId
        try { old.close() } catch (t: Throwable) { logger.warn(TAG, "old.close() threw", t) }
        logger.info(
            TAG,
            "active backend → ${newBackend.spec.modelId} (${newBackend.backendName}, delegate=${newBackend.delegate ?: "—"})",
        )
        BackendSnapshot(
            spec = newBackend.spec,
            backendName = newBackend.backendName,
            delegate = newBackend.delegate,
        )
    }

    fun close() = lock.withLock {
        try { current.close() } catch (_: Throwable) {}
    }

    private fun tryLoadOnnx(spec: ModelSpec): DetectorBackend? {
        val file = loader.resolveOrCopy(spec) ?: run {
            logger.warn(
                TAG,
                "no ${spec.modelFile} found in external/cache/assets for modelId '${spec.modelId}'",
            )
            return null
        }
        return try {
            val hash = try { loader.sha256(file).take(16) } catch (_: Throwable) { null }
            val resolvedSpec = spec.copy(modelHash = hash)
            OnnxRuntimeAndroidBackend(resolvedSpec, file.absolutePath, tryNnapi = true)
                .also { it.warmup() }
        } catch (t: Throwable) {
            logger.error(TAG, "ONNX init failed for '${spec.modelId}'", t)
            null
        }
    }

    companion object {
        const val TAG = "BackendMgr"
    }
}

/**
 * Atomic per-frame output from [AndroidBackendManager.analyze]: the
 * inference result plus a spec snapshot captured under the same lock
 * acquisition, so telemetry built from them cannot disagree.
 */
data class AnalysisOutput(
    val result: InferenceResult,
    val spec: ModelSpec,
)

data class BackendSnapshot(
    val spec: ModelSpec,
    val backendName: String,
    val delegate: String?,
)
