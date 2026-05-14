package io.kitware.shitspotter.core

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/**
 * Shared app state. Both the Android live-camera flow and the desktop
 * still-image harness write into the same state, which is what lets the
 * overlay/HUD/failure-capture composables stay platform-agnostic.
 */
class AppState {
    var lastDetections: List<Detection> by mutableStateOf(emptyList())
        private set
    var lastTelemetry: FrameTelemetry? by mutableStateOf(null)
        private set
    var activeModelId: String by mutableStateOf(ModelRegistry.default.modelId)
    var scoreThreshold: Float by mutableStateOf(ModelRegistry.default.scoreThreshold)
    var showFps: Boolean by mutableStateOf(true)
    var showOverlay: Boolean by mutableStateOf(true)
    var lastFrameWidth: Int by mutableStateOf(0)
        private set
    var lastFrameHeight: Int by mutableStateOf(0)
        private set
    var lastError: String? by mutableStateOf(null)
        private set
    var failureCasesSavedCount: Int by mutableStateOf(0)
    var photosSavedCount: Int by mutableStateOf(0)
    /** Which physical camera to use on Android — back is the default
     *  because shitspotter is a "look at the ground" app. */
    var useFrontCamera: Boolean by mutableStateOf(false)
    var metadataMode: MetadataMode by mutableStateOf(MetadataMode.FULL)
    var reviewMode: Boolean by mutableStateOf(false)
    var capturedPhotos: List<CaptureReviewEntry> by mutableStateOf(emptyList())

    fun pushFrame(detections: List<Detection>, telemetry: FrameTelemetry, frameW: Int, frameH: Int) {
        lastDetections = detections
        lastTelemetry = telemetry
        lastFrameWidth = frameW
        lastFrameHeight = frameH
        lastError = null
    }

    fun setError(message: String?) {
        lastError = message
    }

    fun activeModel(): ModelSpec = ModelRegistry.byId(activeModelId) ?: ModelRegistry.default
}
