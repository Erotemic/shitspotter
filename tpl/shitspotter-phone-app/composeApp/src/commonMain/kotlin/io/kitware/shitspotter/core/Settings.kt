package io.kitware.shitspotter.core

import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString

/**
 * Persisted user preferences. Kept tiny — we only persist the bits that
 * cost the user time to re-set. Everything else (last-detection history,
 * FPS counters, transient state) lives in [AppState] and is regenerated
 * on every launch.
 */
@Serializable
data class AppSettings(
    val activeModelId: String = ModelRegistry.default.modelId,
    val scoreThreshold: Float = ModelRegistry.default.scoreThreshold,
    val showFps: Boolean = true,
    val showOverlay: Boolean = true,
    val useFrontCamera: Boolean = false,
    val metadataMode: MetadataMode = MetadataMode.FULL,
)

interface SettingsStore {
    fun load(): AppSettings
    fun save(settings: AppSettings)
}

/**
 * Trivial in-memory store; useful in tests and as a safety fallback when
 * the platform-specific store fails to initialise.
 */
class InMemorySettingsStore(initial: AppSettings = AppSettings()) : SettingsStore {
    private var current: AppSettings = initial
    override fun load(): AppSettings = current
    override fun save(settings: AppSettings) {
        current = settings
    }
}

object SettingsSerialization {
    fun encode(settings: AppSettings): String =
        FailureCaseSerialization.json.encodeToString(settings)

    fun decode(text: String): AppSettings =
        FailureCaseSerialization.json.decodeFromString<AppSettings>(text)
}

/**
 * Helper that updates [AppState] from a snapshot and produces a new
 * snapshot from the current state. Lets the platform store be a thin
 * load-and-write wrapper without owning any AppState lifecycle.
 */
fun AppState.applySettings(s: AppSettings) {
    activeModelId = s.activeModelId
    scoreThreshold = s.scoreThreshold
    showFps = s.showFps
    showOverlay = s.showOverlay
    useFrontCamera = s.useFrontCamera
    metadataMode = s.metadataMode
}

fun AppState.toSettings(): AppSettings = AppSettings(
    activeModelId = activeModelId,
    scoreThreshold = scoreThreshold,
    showFps = showFps,
    showOverlay = showOverlay,
    useFrontCamera = useFrontCamera,
    metadataMode = metadataMode,
)
