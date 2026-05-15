package io.github.erotemic.shitspotter.desktop

import io.github.erotemic.shitspotter.core.AppSettings
import io.github.erotemic.shitspotter.core.PrintlnLogger
import io.github.erotemic.shitspotter.core.SettingsSerialization
import io.github.erotemic.shitspotter.core.SettingsStore
import java.io.File

/**
 * Desktop settings store: a single JSON file at
 * `<workdir>/.shitspotter/settings.json`. Errors fall back to defaults
 * with a logged warning rather than crashing the harness.
 */
class FileSettingsStore(
    private val file: File = File(System.getProperty("user.home"), ".shitspotter/settings.json"),
) : SettingsStore {
    init {
        file.parentFile?.takeUnless { it.isDirectory }?.mkdirs()
    }

    override fun load(): AppSettings {
        if (!file.isFile) return AppSettings()
        return try {
            SettingsSerialization.decode(file.readText())
        } catch (t: Throwable) {
            PrintlnLogger.warn("ShitSpotter.Settings", "could not load ${file.path}", t)
            AppSettings()
        }
    }

    override fun save(settings: AppSettings) {
        try {
            file.writeText(SettingsSerialization.encode(settings))
        } catch (t: Throwable) {
            PrintlnLogger.warn("ShitSpotter.Settings", "could not save ${file.path}", t)
        }
    }
}
