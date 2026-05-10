package io.kitware.shitspotter.desktop

import io.kitware.shitspotter.core.FrameSource
import java.io.File
import java.util.concurrent.atomic.AtomicInteger

/**
 * Replays a directory of still images as a synthetic "video stream"
 * for the desktop harness. Files are sorted lexicographically (so
 * `0001.jpg`, `0002.jpg`, `0003.jpg` work as you'd expect) and looped
 * forever via [next].
 *
 * This is intentionally simple — we don't pull in JavaCV / ffmpeg-java
 * just to validate the desktop pipeline. If a real MP4 path becomes
 * necessary we can add it as a separate source.
 */
class FrameDirectorySource(directory: File, exts: Set<String> = DEFAULT_EXTS) {
    private val files: List<File>
    private val cursor = AtomicInteger(0)

    init {
        require(directory.isDirectory) { "Not a directory: ${directory.absolutePath}" }
        files = directory.listFiles { f -> f.isFile && exts.contains(f.extension.lowercase()) }
            ?.sortedBy { it.name }
            ?: emptyList()
        require(files.isNotEmpty()) { "No image files in $directory (extensions: $exts)" }
    }

    val frameCount: Int get() = files.size

    fun next(): FrameSource {
        val idx = cursor.getAndIncrement() % files.size
        return StillImageFrameSource.fromFile(files[idx])
    }

    fun reset() {
        cursor.set(0)
    }

    companion object {
        val DEFAULT_EXTS: Set<String> = setOf("jpg", "jpeg", "png", "bmp")
    }
}
