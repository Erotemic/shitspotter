package io.github.erotemic.shitspotter.core

import kotlinx.serialization.Serializable

@Serializable
enum class CaptureLabel { TRUE_POSITIVE, FALSE_POSITIVE, FALSE_NEGATIVE, TRUE_NEGATIVE, UNCERTAIN }

@Serializable
enum class MetadataMode { FULL, NO_GPS, NONE }

/** Per-detection annotation applied by the user in the review viewer. */
@Serializable
enum class DetectionAnnotation { TRUE_POSITIVE, FALSE_POSITIVE }

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
    /** Pixel dimensions of the camera analysis frame (detection coordinate space). */
    val frameWidth: Int? = null,
    val frameHeight: Int? = null,
    /** Map of detection index (as string) → TP/FP annotation set by the user. */
    val detectionAnnotations: Map<String, DetectionAnnotation> = emptyMap(),
    /** User-drawn bounding boxes marking missed detections (FN), in frame pixel space. */
    val missedBoxes: List<BoundingBox> = emptyList(),
)

data class CaptureReviewEntry(
    val filePath: String,
    val timestamp: String,
    val label: CaptureLabel,
    val detectionCount: Int,
    val note: String? = null,
)
