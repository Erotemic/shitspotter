package io.kitware.shitspotter.core

import kotlinx.serialization.Serializable

@Serializable
enum class CaptureLabel { TRUE_POSITIVE, FALSE_POSITIVE, FALSE_NEGATIVE, TRUE_NEGATIVE, UNCERTAIN }

@Serializable
enum class MetadataMode { FULL, NO_GPS, NONE }

@Serializable
data class CaptureMetadata(
    val timestamp: String,
    val label: CaptureLabel,
    val modelId: String,
    val modelHash: String?,
    val appCommit: String,
    val buildDate: String,
    val deviceModel: String,
    val scoreThreshold: Float,
    val detections: List<Detection>,
    val latitude: Double? = null,
    val longitude: Double? = null,
    val userNote: String? = null,
)

data class CaptureReviewEntry(
    val filePath: String,
    val timestamp: String,
    val label: CaptureLabel,
    val detectionCount: Int,
    val note: String? = null,
)
