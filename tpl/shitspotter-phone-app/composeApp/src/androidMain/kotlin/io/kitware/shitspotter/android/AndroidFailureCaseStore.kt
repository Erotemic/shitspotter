package io.kitware.shitspotter.android

import android.content.Context
import io.kitware.shitspotter.core.FailureCaseMetadata
import io.kitware.shitspotter.core.FailureCaseSerialization
import io.kitware.shitspotter.core.FailureCaseStore
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

class AndroidFailureCaseStore(
    context: Context,
    rootDir: File = File(context.getExternalFilesDir(null) ?: context.filesDir, "failure_cases"),
) : FailureCaseStore {
    private val root: File = rootDir.apply { mkdirs() }

    override fun save(imageJpegBytes: ByteArray, metadata: FailureCaseMetadata): String {
        val ts = metadata.timestamp.replace(":", "").replace("-", "").replace(".", "_")
        val dir = File(root, ts).apply { mkdirs() }
        File(dir, "image.jpg").writeBytes(imageJpegBytes)
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
