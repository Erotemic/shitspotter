package io.kitware.shitspotter.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SettingsTest {

    @Test
    fun default_settings_match_registry_default() {
        val s = AppSettings()
        assertEquals(ModelRegistry.default.modelId, s.activeModelId)
        assertEquals(ModelRegistry.default.scoreThreshold, s.scoreThreshold)
        assertTrue(s.showFps)
        assertTrue(s.showOverlay)
    }

    @Test
    fun encode_decode_round_trip() {
        val s = AppSettings(
            activeModelId = "yolox-nano-poop-cropped-v1",
            scoreThreshold = 0.42f,
            showFps = false,
            showOverlay = true,
        )
        val text = SettingsSerialization.encode(s)
        val back = SettingsSerialization.decode(text)
        assertEquals(s, back)
    }

    @Test
    fun in_memory_store_round_trips() {
        val store = InMemorySettingsStore()
        val s = AppSettings(activeModelId = "stub-fake-detector", scoreThreshold = 0.1f)
        store.save(s)
        assertEquals(s, store.load())
    }

    @Test
    fun applySettings_then_toSettings_round_trips() {
        val state = AppState()
        val s = AppSettings(
            activeModelId = "yolox-nano-poop-cropped-v1",
            scoreThreshold = 0.33f,
            showFps = false,
            showOverlay = false,
        )
        state.applySettings(s)
        assertEquals(s, state.toSettings())
    }
}
