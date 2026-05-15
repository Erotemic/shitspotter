package io.github.erotemic.shitspotter.android

import android.content.Context
import android.graphics.ImageFormat
import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import io.github.erotemic.shitspotter.core.AppLogger
import io.github.erotemic.shitspotter.core.AppState
import io.github.erotemic.shitspotter.core.BoundingBox
import io.github.erotemic.shitspotter.core.BuildInfo
import io.github.erotemic.shitspotter.core.FpsCounter
import io.github.erotemic.shitspotter.core.FrameSource
import io.github.erotemic.shitspotter.core.FrameTelemetry
import io.github.erotemic.shitspotter.core.LatencyAccumulator
import io.github.erotemic.shitspotter.core.filterByScore
import io.github.erotemic.shitspotter.core.nowMonoMs
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/**
 * Wires CameraX [ImageAnalysis] into a [DetectorBackend]. KEEP_ONLY_LATEST is
 * the only acceptable strategy here — we never queue stale frames behind a
 * slow detector. Every ImageProxy is closed exactly once.
 */
class CameraAnalysisLoop(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val state: AppState,
    private val backendManager: AndroidBackendManager,
    private val logger: AppLogger = AndroidLogger,
    private val targetAnalysisSize: Size = Size(640, 480),
) {
    @Volatile var isPaused: Boolean = false
    @Volatile private var droppedFrames: AtomicLong = AtomicLong(0L)
    private var lastSelectorWasFront: Boolean = false
    private var rebindFn: (() -> Unit)? = null
    @Volatile private var cameraRef: Camera? = null
    private var imageCaptureUseCase: ImageCapture? = null

    private val analyzerExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "shitspotter-analyzer").apply { priority = Thread.NORM_PRIORITY + 1 }
    }
    @Volatile private var closed = false
    @Volatile private var cameraProvider: ProcessCameraProvider? = null
    private val fpsCounter = FpsCounter(windowMs = 1500L)
    private val captureLat = LatencyAccumulator(60)
    private val preLat = LatencyAccumulator(60)
    private val infLat = LatencyAccumulator(60)
    private val postLat = LatencyAccumulator(60)
    private val overlayLat = LatencyAccumulator(60)

    fun bind(
        previewSurfaceProvider: Preview.SurfaceProvider,
    ): ListenableFuture<ProcessCameraProvider> {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            // The activity may have called close() while the provider
            // future was still pending. Don't bind a doomed loop.
            if (closed) return@addListener
            try {
                val provider = future.get()
                cameraProvider = provider
                val rebind: () -> Unit = rebind@{
                    if (closed) return@rebind
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewSurfaceProvider)
                    }
                    val analysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setResolutionSelector(
                            ResolutionSelector.Builder()
                                .setResolutionStrategy(
                                    ResolutionStrategy(
                                        targetAnalysisSize,
                                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                                    ),
                                )
                                .build(),
                        )
                        .build()
                    analysis.setAnalyzer(analyzerExecutor) { proxy -> handleFrame(proxy) }

                    val imgCap = ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                        .build()
                    imageCaptureUseCase = imgCap

                    val selector = if (state.useFrontCamera) {
                        CameraSelector.DEFAULT_FRONT_CAMERA
                    } else {
                        CameraSelector.DEFAULT_BACK_CAMERA
                    }
                    provider.unbindAll()
                    val cam = provider.bindToLifecycle(lifecycleOwner, selector, preview, analysis, imgCap)
                    cameraRef = cam
                    lastSelectorWasFront = state.useFrontCamera
                    logger.info(
                        TAG,
                        "CameraX bound; analysis target=$targetAnalysisSize selector=" +
                            if (state.useFrontCamera) "front" else "back",
                    )
                }
                rebindFn = rebind
                rebind()
            } catch (t: Throwable) {
                logger.error(TAG, "CameraX bind failed", t)
                state.setError("Camera bind failed: ${t.message}")
            }
        }, ContextCompatExecutors.main(context))
        return future
    }

    /** Tear down the CameraX bindings and shut down the analyzer thread.
     *  Idempotent. Call from `MainActivity.onDestroy` (or via Compose's
     *  `DisposableEffect`) so the singleThreadExecutor doesn't leak. */
    fun close() {
        if (closed) return
        closed = true
        try {
            ContextCompatExecutors.main(context).execute {
                try { cameraProvider?.unbindAll() } catch (_: Throwable) {}
            }
        } catch (_: Throwable) {}
        analyzerExecutor.shutdown()
        try {
            analyzerExecutor.awaitTermination(1, TimeUnit.SECONDS)
        } catch (_: Throwable) {}
        if (!analyzerExecutor.isTerminated) analyzerExecutor.shutdownNow()
        logger.info(TAG, "CameraAnalysisLoop closed")
    }

    /** Rebind CameraX with the current `state.useFrontCamera` value if it
     *  has changed since the last bind. Safe to call from the main thread
     *  on every frame; no-op when nothing changed. */
    fun rebindIfCameraChanged() {
        if (state.useFrontCamera != lastSelectorWasFront) {
            ContextCompatExecutors.main(context).execute {
                rebindFn?.invoke()
            }
        }
    }

    @Volatile var lastAnalyzedFrame: ImageProxyFrame? = null
        private set

    private fun handleFrame(proxy: ImageProxy) {
        try {
            if (closed) return
            // Detect a state-driven camera switch and re-bind on the
            // main thread before the next frame arrives. Cheap when
            // there's no change.
            rebindIfCameraChanged()
            if (isPaused) {
                droppedFrames.incrementAndGet()
                return
            }
            val captureStart = nowMonoMs()
            val frame = ImageProxyFrame.from(proxy)
            lastAnalyzedFrame = frame
            captureLat.record(nowMonoMs() - captureStart)

            // Runs under the backend manager's lock — concurrent
            // setActive(...) cannot close the backend mid-analyse, and
            // the returned spec is captured atomically alongside the
            // result so the HUD never reports a stale model id.
            val out = backendManager.analyze(frame)
            val result = out.result
            val spec = out.spec

            preLat.record(result.preprocessMs)
            infLat.record(result.inferenceMs)
            postLat.record(result.postprocessMs)

            val overlayStart = nowMonoMs()
            val nowMs = System.currentTimeMillis()
            val fps = fpsCounter.mark(nowMs)
            val overlayMs = nowMonoMs() - overlayStart
            overlayLat.record(overlayMs)

            val raw = result.detections.filterByScore(state.scoreThreshold)
            // Compose the detector's frame-pixel coordinates with the camera
            // rotation so the overlay can draw boxes in display orientation
            // without knowing about CameraX rotationDegrees itself. We
            // compute the rotated frame dims from a dummy box so the result
            // is consistent whether or not we have any detections.
            val rotation = frame.rotationDegrees
            val frameRotated = BoundingBox(0f, 0f, 0f, 0f)
                .rotated(rotation, frame.width, frame.height)
            val rotatedW = frameRotated.second
            val rotatedH = frameRotated.third
            val filtered = raw.map { d ->
                val r = d.box.rotated(rotation, frame.width, frame.height)
                d.copy(box = r.first)
            }
            val telemetry = FrameTelemetry(
                deviceModel = BuildInfo.deviceModel,
                osVersion = BuildInfo.osVersion,
                appCommit = BuildInfo.appCommit,
                modelId = spec.modelId,
                modelHash = spec.modelHash,
                runtimeBackend = result.backendName,
                delegate = result.delegate,
                inputWidth = spec.inputWidth,
                inputHeight = spec.inputHeight,
                captureMs = captureLat.mean(),
                preprocessMs = result.preprocessMs,
                inferenceMs = result.inferenceMs,
                postprocessMs = result.postprocessMs,
                overlayMs = overlayMs,
                fpsRecent = fps,
                detectionCount = filtered.size,
                droppedFrames = droppedFrames.get(),
            )
            state.pushFrame(filtered, telemetry, rotatedW, rotatedH)
        } catch (t: Throwable) {
            logger.error(TAG, "frame analysis failed", t)
            state.setError(t.message ?: t::class.simpleName ?: "unknown")
        } finally {
            proxy.close()
        }
    }

    fun setTorch(enabled: Boolean) {
        cameraRef?.cameraControl?.enableTorch(enabled)
    }

    fun takePicture(
        outputFile: java.io.File,
        executor: java.util.concurrent.Executor,
        onSuccess: (java.io.File) -> Unit,
        onError: (Exception) -> Unit,
    ) {
        val cap = imageCaptureUseCase ?: run {
            onError(IllegalStateException("camera not ready"))
            return
        }
        val opts = ImageCapture.OutputFileOptions.Builder(outputFile).build()
        cap.takePicture(opts, executor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(output: ImageCapture.OutputFileResults) = onSuccess(outputFile)
            override fun onError(exc: ImageCaptureException) = onError(exc)
        })
    }

    companion object {
        const val TAG = "AnalysisLoop"
    }
}

/** Wraps a CameraX ImageProxy as a [FrameSource]. We only convert to RGB888
 * lazily — backends that don't need RGB skip the copy. The raw RGBA bytes
 * are also exposed so that the failure-case capture can encode the latest
 * analyzed frame as JPEG without re-grabbing it from the camera. */
class ImageProxyFrame private constructor(
    override val width: Int,
    override val height: Int,
    override val rotationDegrees: Int,
    val rgbaPlane: ByteArray,
    val rowStride: Int,
    val pixelStride: Int,
) : FrameSource {

    override fun toRgb888(): ByteArray {
        if (rowStride == width * pixelStride && pixelStride == 4) {
            // Tight RGBA8888 — reduce in place.
            val out = ByteArray(width * height * 3)
            var src = 0
            var dst = 0
            val total = width * height
            for (i in 0 until total) {
                out[dst] = rgbaPlane[src]
                out[dst + 1] = rgbaPlane[src + 1]
                out[dst + 2] = rgbaPlane[src + 2]
                src += 4
                dst += 3
            }
            return out
        }
        // Fallback: row-by-row stride-aware copy.
        val out = ByteArray(width * height * 3)
        var dst = 0
        for (y in 0 until height) {
            val rowStart = y * rowStride
            for (x in 0 until width) {
                val src = rowStart + x * pixelStride
                out[dst] = rgbaPlane[src]
                out[dst + 1] = rgbaPlane[src + 1]
                out[dst + 2] = rgbaPlane[src + 2]
                dst += 3
            }
        }
        return out
    }

    /** Encode this frame as JPEG via Android Bitmap → JPEG. Used by the
     *  failure-case capture path. Honours [rotationDegrees] so the saved
     *  JPEG matches what the user saw on screen. */
    fun encodeJpeg(quality: Int = 85): ByteArray {
        val argb = IntArray(width * height)
        if (rowStride == width * pixelStride && pixelStride == 4) {
            var src = 0
            for (i in argb.indices) {
                val r = rgbaPlane[src].toInt() and 0xFF
                val g = rgbaPlane[src + 1].toInt() and 0xFF
                val b = rgbaPlane[src + 2].toInt() and 0xFF
                val a = rgbaPlane[src + 3].toInt() and 0xFF
                argb[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
                src += 4
            }
        } else {
            var idx = 0
            for (y in 0 until height) {
                val rowStart = y * rowStride
                for (x in 0 until width) {
                    val src = rowStart + x * pixelStride
                    val r = rgbaPlane[src].toInt() and 0xFF
                    val g = rgbaPlane[src + 1].toInt() and 0xFF
                    val b = rgbaPlane[src + 2].toInt() and 0xFF
                    val a = if (pixelStride >= 4) rgbaPlane[src + 3].toInt() and 0xFF else 0xFF
                    argb[idx++] = (a shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
        val bmp = android.graphics.Bitmap.createBitmap(argb, width, height, android.graphics.Bitmap.Config.ARGB_8888)
        val rotated = if (rotationDegrees != 0) {
            val m = android.graphics.Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            android.graphics.Bitmap.createBitmap(bmp, 0, 0, width, height, m, true).also { bmp.recycle() }
        } else bmp
        val baos = java.io.ByteArrayOutputStream(width * height / 8)
        rotated.compress(android.graphics.Bitmap.CompressFormat.JPEG, quality, baos)
        rotated.recycle()
        return baos.toByteArray()
    }

    companion object {
        fun from(proxy: ImageProxy): ImageProxyFrame {
            val plane = proxy.planes[0]
            val buf = plane.buffer
            val bytes = ByteArray(buf.remaining())
            buf.get(bytes)
            return ImageProxyFrame(
                width = proxy.width,
                height = proxy.height,
                rotationDegrees = proxy.imageInfo.rotationDegrees,
                rgbaPlane = bytes,
                rowStride = plane.rowStride,
                pixelStride = plane.pixelStride,
            )
        }
    }
}

/** Tiny shim around ContextCompat.getMainExecutor with a stable name. */
internal object ContextCompatExecutors {
    fun main(context: Context): Executor =
        androidx.core.content.ContextCompat.getMainExecutor(context)
}
