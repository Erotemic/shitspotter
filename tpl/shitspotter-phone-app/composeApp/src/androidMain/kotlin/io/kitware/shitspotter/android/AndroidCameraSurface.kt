package io.kitware.shitspotter.android

import android.content.Context
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.LifecycleOwner
import io.kitware.shitspotter.core.AppState
import io.kitware.shitspotter.ui.CameraSurface

class AndroidCameraSurface(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val state: AppState,
    private val backendManager: AndroidBackendManager,
) : CameraSurface {

    private var loop: CameraAnalysisLoop? = null

    fun setPaused(paused: Boolean) {
        loop?.isPaused = paused
    }

    /** True iff at least one camera frame has been received and wrapped. */
    fun hasAnalyzedFrame(): Boolean = loop?.lastAnalyzedFrame != null

    fun encodeLastFrameAsJpeg(quality: Int = 85): ByteArray =
        loop?.lastAnalyzedFrame?.encodeJpeg(quality) ?: ByteArray(0)

    /** Tear down the analyzer thread + CameraX bindings. Idempotent. */
    fun close() {
        loop?.close()
        loop = null
    }

    @Composable
    override fun Render(modifier: Modifier) {
        val androidContext = LocalContext.current
        AndroidView(
            modifier = modifier,
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }
                val analysisLoop = CameraAnalysisLoop(
                    context = androidContext,
                    lifecycleOwner = lifecycleOwner,
                    state = state,
                    backendManager = backendManager,
                )
                this.loop = analysisLoop
                analysisLoop.bind(previewView.surfaceProvider)
                previewView
            },
        )
    }
}
