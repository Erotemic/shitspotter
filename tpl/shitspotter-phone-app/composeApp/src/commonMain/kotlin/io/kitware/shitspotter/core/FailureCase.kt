package io.kitware.shitspotter.core

import kotlinx.serialization.Serializable

@Serializable
enum class FailureType {
    FALSE_POSITIVE,
    FALSE_NEGATIVE,
    BAD_LOCALIZATION,
    LAG,
    CRASH,
    UNCERTAIN,
    OTHER,
}

@Serializable
data class FailureCaseMetadata(
    val timestamp: String,
    val deviceModel: String,
    val osVersion: String,
    val appCommit: String,
    val modelId: String,
    val modelHash: String?,
    val runtimeBackend: String,
    val delegate: String?,
    val inputWidth: Int,
    val inputHeight: Int,
    val scoreThreshold: Float,
    val iouThreshold: Float,
    val latencyMs: Double,
    val fpsRecent: Double,
    val failureType: FailureType,
    val userNote: String? = null,
    val detections: List<Detection>,
)
