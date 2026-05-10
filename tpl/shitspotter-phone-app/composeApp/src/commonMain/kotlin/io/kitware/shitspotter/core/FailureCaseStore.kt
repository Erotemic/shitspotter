package io.kitware.shitspotter.core

import kotlinx.serialization.json.Json

/**
 * Platform-agnostic interface to persist a failure case (image bytes +
 * JSON metadata + detections JSON + optional user note) to a folder on
 * the device or desktop disk. Each platform supplies a tiny adapter.
 */
interface FailureCaseStore {
    /** Returns the directory the failure case was written to, for surfacing in the UI. */
    fun save(
        imageJpegBytes: ByteArray,
        metadata: FailureCaseMetadata,
    ): String
}

object FailureCaseSerialization {
    val json: Json = Json {
        prettyPrint = true
        encodeDefaults = true
        ignoreUnknownKeys = true
    }
}
