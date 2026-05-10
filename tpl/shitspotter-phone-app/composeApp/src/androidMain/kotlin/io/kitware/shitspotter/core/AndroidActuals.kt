package io.kitware.shitspotter.core

import io.kitware.shitspotter.BuildConfig
import android.os.Build

actual fun nowMonoMs(): Double = System.nanoTime() / 1_000_000.0

actual object BuildInfo {
    actual val deviceModel: String = "${Build.MANUFACTURER} ${Build.MODEL}"
    actual val osVersion: String = "Android ${Build.VERSION.RELEASE} (sdk=${Build.VERSION.SDK_INT})"
    actual val appCommit: String = BuildConfig.APP_GIT_COMMIT
}
