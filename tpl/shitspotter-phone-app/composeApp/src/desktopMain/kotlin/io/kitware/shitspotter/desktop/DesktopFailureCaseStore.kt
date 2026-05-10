package io.kitware.shitspotter.desktop

import io.kitware.shitspotter.core.FailureCaseMetadata
import io.kitware.shitspotter.core.FailureCaseSerialization
import io.kitware.shitspotter.core.FailureCaseStore
import kotlinx.serialization.encodeToString
import java.io.File

class DesktopFailureCaseStore(
    private val rootDir: File,
) : FailureCaseStore {
    init { rootDir.mkdirs() }

    override fun save(imageJpegBytes: ByteArray, metadata: FailureCaseMetadata): String {
        val ts = metadata.timestamp.replace(":", "").replace("-", "").replace(".", "_")
        val dir = File(rootDir, ts).apply { mkdirs() }
        if (imageJpegBytes.isNotEmpty()) {
            File(dir, "image.jpg").writeBytes(imageJpegBytes)
        }
        File(dir, "metadata.json").writeText(
            FailureCaseSerialization.json.encodeToString(metadata)
        )
        File(dir, "detections.json").writeText(
            FailureCaseSerialization.json.encodeToString(metadata.detections)
        )
        metadata.userNote?.let { File(dir, "user_note.txt").writeText(it) }
        return dir.absolutePath
    }
}
