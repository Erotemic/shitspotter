package io.github.erotemic.shitspotter.desktop

import io.github.erotemic.shitspotter.core.AppSettings
import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals

class FileSettingsStoreTest {

    @Test
    fun load_returns_defaults_when_no_file() {
        val tmp = createTempDirectory("ssp-set").toFile()
        try {
            val store = FileSettingsStore(File(tmp, "settings.json"))
            assertEquals(AppSettings(), store.load())
        } finally {
            tmp.deleteRecursively()
        }
    }

    @Test
    fun save_then_load_round_trips() {
        val tmp = createTempDirectory("ssp-set").toFile()
        try {
            val file = File(tmp, "settings.json")
            val store = FileSettingsStore(file)
            val s = AppSettings(
                activeModelId = "yolox-nano-poop-cropped-v1",
                scoreThreshold = 0.42f,
                showFps = false,
                showOverlay = true,
            )
            store.save(s)
            val loaded = FileSettingsStore(file).load()
            assertEquals(s, loaded)
        } finally {
            tmp.deleteRecursively()
        }
    }

    @Test
    fun load_falls_back_to_defaults_on_corrupt_json() {
        val tmp = createTempDirectory("ssp-set").toFile()
        try {
            val file = File(tmp, "settings.json")
            file.writeText("{this isn't json")
            val loaded = FileSettingsStore(file).load()
            assertEquals(AppSettings(), loaded)
        } finally {
            tmp.deleteRecursively()
        }
    }
}
