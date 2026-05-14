package io.kitware.shitspotter.core

actual fun nowMonoMs(): Double = System.nanoTime() / 1_000_000.0

actual object BuildInfo {
    actual val deviceModel: String = "Linux desktop (${System.getProperty("os.name")} ${System.getProperty("os.arch")})"
    actual val osVersion: String = System.getProperty("os.version") ?: "unknown"
    actual val appCommit: String = System.getProperty("ssp.app.commit") ?: "dev"
    actual val buildDate: String = System.getProperty("ssp.app.build.date") ?: "dev"
}
