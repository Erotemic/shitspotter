package io.github.erotemic.shitspotter.android

import android.content.Context
import io.github.erotemic.shitspotter.core.ModelSpec
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest

/**
 * Resolves a [ModelSpec] to an absolute file path the ONNX backend can open.
 *
 * Lookup order:
 *   1. external files dir: <app>/files/models/<modelFile> (sideload-friendly)
 *   2. internal cache, if previously copied from assets
 *   3. APK assets/<modelFile> — copied to the cache once, then reused
 *
 * If none of the above resolves, returns null and the caller falls back to
 * the stub detector.
 */
class AndroidModelLoader(private val context: Context) {

    fun resolveOrCopy(spec: ModelSpec): File? {
        val externalDir = File(
            context.getExternalFilesDir(null) ?: context.filesDir,
            "models",
        ).apply { mkdirs() }
        val externalFile = File(externalDir, spec.modelFile)
        if (externalFile.isFile) return externalFile

        val cacheDir = File(context.cacheDir, "models").apply { mkdirs() }
        val cacheFile = File(cacheDir, spec.modelFile)
        if (cacheFile.isFile && cacheFile.length() > 0L) return cacheFile

        // Try copying from APK assets.
        return try {
            context.assets.open(spec.modelFile).use { input ->
                FileOutputStream(cacheFile).use { output ->
                    input.copyTo(output, bufferSize = 1 shl 16)
                }
            }
            if (cacheFile.length() > 0L) cacheFile else null
        } catch (_: Throwable) {
            null
        }
    }

    fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { ins ->
            val buf = ByteArray(1 shl 15)
            while (true) {
                val n = ins.read(buf)
                if (n <= 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString(separator = "") { b -> "%02x".format(b) }
    }
}
