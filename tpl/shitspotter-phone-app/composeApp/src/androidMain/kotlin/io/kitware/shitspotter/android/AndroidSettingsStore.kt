package io.kitware.shitspotter.android

import android.content.Context
import io.kitware.shitspotter.core.AppSettings
import io.kitware.shitspotter.core.SettingsSerialization
import io.kitware.shitspotter.core.SettingsStore

/**
 * Persists [AppSettings] via SharedPreferences using a single JSON
 * blob — keeps the on-device serialization format identical to the
 * desktop file store and to whatever future agents choose to read off
 * the failure-case dumps.
 */
class AndroidSettingsStore(context: Context) : SettingsStore {
    private val prefs = context.getSharedPreferences(PREF_FILE, Context.MODE_PRIVATE)

    override fun load(): AppSettings {
        val raw = prefs.getString(KEY, null) ?: return AppSettings()
        return try {
            SettingsSerialization.decode(raw)
        } catch (t: Throwable) {
            AppSettings()
        }
    }

    override fun save(settings: AppSettings) {
        prefs.edit().putString(KEY, SettingsSerialization.encode(settings)).apply()
    }

    companion object {
        private const val PREF_FILE = "io.kitware.shitspotter.settings"
        private const val KEY = "settings_json"
    }
}
