package io.kitware.shitspotter.android

import android.content.Context
import io.kitware.shitspotter.core.AppLogger
import io.kitware.shitspotter.core.DetectorBackend
import io.kitware.shitspotter.core.ModelFormat
import io.kitware.shitspotter.core.ModelRegistry
import io.kitware.shitspotter.core.ModelSpec
import io.kitware.shitspotter.core.PrintlnLogger
import io.kitware.shitspotter.core.StubDetectorBackend
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Owns the current Android [DetectorBackend] and switches it when the
 * active model id changes. Synchronous on the request thread so a
 * camera-frame callback never sees a half-swapped backend.
 *
 * Backend lifecycle:
 *   - Stub is the fallback when no ONNX file is reachable or load fails.
 *   - Real ONNX backends are warmed up before becoming "current".
 *   - The previous backend is `close()`-d after the new one is live so
 *     `analyze()` callers see a continuous backend reference (never
 *     null, never closed).
 *
 * Thread safety: a single lock guards [setActive] and [current]. The
 * analysis loop calls [current] on every frame; CompareCli and the UI
 * chip row call [setActive] from non-frame threads.
 */
class AndroidBackendManager(
    private val context: Context,
    private val logger: AppLogger = PrintlnLogger,
) {
    private val lock = ReentrantLock()
    private val loader = AndroidModelLoader(context)
    @Volatile private var _current: DetectorBackend = StubDetectorBackend()
    @Volatile private var _currentId: String = _current.spec.modelId

    val current: DetectorBackend get() = _current
    val currentModelId: String get() = _currentId

    /**
     * Switch to the model registered as [modelId]. Returns the new backend
     * (which may be a stub if the id is unknown or the model fails to load).
     * Idempotent — calling with the already-active id is a no-op.
     */
    fun setActive(modelId: String): DetectorBackend = lock.withLock {
        if (modelId == _currentId) return@withLock _current
        val spec = ModelRegistry.byId(modelId)
        val newBackend = when {
            spec == null -> {
                logger.warn(
                    TAG,
                    "unknown modelId '$modelId'; staying on ${_currentId}",
                )
                return@withLock _current
            }
            spec.format == ModelFormat.STUB -> StubDetectorBackend(spec)
            spec.format == ModelFormat.ONNX -> tryLoadOnnx(spec)
                ?: StubDetectorBackend(spec)
            else -> {
                logger.warn(
                    TAG,
                    "unsupported format ${spec.format} for modelId '$modelId'; falling back to stub",
                )
                StubDetectorBackend(spec)
            }
        }
        val old = _current
        _current = newBackend
        _currentId = modelId
        // Close the old backend AFTER the swap so any in-flight analyze()
        // call that already captured the old reference can finish.
        try { old.close() } catch (t: Throwable) { logger.warn(TAG, "old.close() threw", t) }
        logger.info(
            TAG,
            "active backend → ${newBackend.spec.modelId} (${newBackend.backendName}, delegate=${newBackend.delegate ?: "—"})",
        )
        newBackend
    }

    fun close() = lock.withLock {
        try { _current.close() } catch (_: Throwable) {}
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
        const val TAG = "ShitSpotter.BackendMgr"
    }
}
