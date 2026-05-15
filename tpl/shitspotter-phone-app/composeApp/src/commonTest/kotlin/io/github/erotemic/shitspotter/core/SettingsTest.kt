package io.github.erotemic.shitspotter.core

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
            useFrontCamera = true,
        )
        val text = SettingsSerialization.encode(s)
        val back = SettingsSerialization.decode(text)
        assertEquals(s, back)
    }

    @Test
    fun decode_ignores_unknown_fields() {
        // A future settings file with extra fields should still decode
        // cleanly into the current schema (forward-compat).
        val text = """
        {
            "activeModelId": "stub-fake-detector",
            "scoreThreshold": 0.0,
            "showFps": true,
            "showOverlay": true,
            "useFrontCamera": false,
            "futureFieldThatDoesNotExistYet": "ignored"
        }
        """.trimIndent()
        val s = SettingsSerialization.decode(text)
        assertEquals("stub-fake-detector", s.activeModelId)
    }

    @Test
    fun decode_falls_back_to_defaults_for_missing_fields() {
        // Older settings JSON (before useFrontCamera was added) should
        // still decode by picking up the default for the missing field.
        val text = """
        {
            "activeModelId": "stub-fake-detector",
            "scoreThreshold": 0.0,
            "showFps": true,
            "showOverlay": true
        }
        """.trimIndent()
        val s = SettingsSerialization.decode(text)
        assertEquals("stub-fake-detector", s.activeModelId)
        // Default for the missing field.
        assertEquals(false, s.useFrontCamera)
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
