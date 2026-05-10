package io.kitware.shitspotter.core

import kotlinx.serialization.Serializable

@Serializable
enum class ModelFormat { ONNX, TFLITE, PTE, COREML, STUB }

@Serializable
enum class InputLayout { NCHW, NHWC }

@Serializable
enum class ColorOrder { RGB, BGR, YUV }

@Serializable
enum class ResizePolicy { LETTERBOX, STRETCH, CENTER_CROP }

@Serializable
enum class PostprocessType {
    YOLOX,
    YOLO_V9,
    GENERIC_BOX_SCORE_CLASS,
    NONE,
    STUB,
}

@Serializable
data class Normalization(
    val mean: List<Float> = listOf(0f, 0f, 0f),
    val std: List<Float> = listOf(1f, 1f, 1f),
    val scale: Float = 1f,
)

@Serializable
data class ModelSpec(
    val modelId: String,
    val displayName: String,
    val modelFile: String,
    val format: ModelFormat,
    val inputWidth: Int,
    val inputHeight: Int,
    val inputLayout: InputLayout,
    val colorOrder: ColorOrder,
    val normalization: Normalization,
    val resizePolicy: ResizePolicy,
    val postprocessType: PostprocessType,
    val classNames: List<String>,
    val scoreThreshold: Float = 0.25f,
    val iouThreshold: Float = 0.45f,
    val modelHash: String? = null,
    val modelVersion: String? = null,
    val trainingDatasetHint: String? = null,
    val notes: String? = null,
) {
    companion object {
        val STUB = ModelSpec(
            modelId = "stub-fake-detector",
            displayName = "Stub (no inference)",
            modelFile = "<none>",
            format = ModelFormat.STUB,
            inputWidth = 640,
            inputHeight = 640,
            inputLayout = InputLayout.NCHW,
            colorOrder = ColorOrder.RGB,
            normalization = Normalization(scale = 1f / 255f),
            resizePolicy = ResizePolicy.LETTERBOX,
            postprocessType = PostprocessType.STUB,
            classNames = listOf("poop"),
            scoreThreshold = 0.0f,
            notes = "Returns a deterministic fake box for Milestone 1 wiring tests.",
        )

        val YOLOX_NANO_POOP = ModelSpec(
            modelId = "yolox-nano-poop-cropped-v1",
            displayName = "YOLOX-nano poop (cropped, v1)",
            modelFile = "yolox_nano_poop_cropped_only_best.onnx",
            format = ModelFormat.ONNX,
            inputWidth = 416,
            inputHeight = 416,
            inputLayout = InputLayout.NCHW,
            colorOrder = ColorOrder.RGB,
            normalization = Normalization(scale = 1f),
            resizePolicy = ResizePolicy.LETTERBOX,
            postprocessType = PostprocessType.YOLOX,
            classNames = listOf("poop"),
            scoreThreshold = 0.25f,
            iouThreshold = 0.45f,
            notes = "First real model. Reference ONNX from tpl/poop_models/. " +
                "Input expected as raw RGB uint8 → float (no /255 in YOLOX preprocessing).",
        )
    }
}

object ModelRegistry {
    val all: List<ModelSpec> = listOf(
        ModelSpec.STUB,
        ModelSpec.YOLOX_NANO_POOP,
    )

    fun byId(id: String): ModelSpec? = all.firstOrNull { it.modelId == id }

    val default: ModelSpec = ModelSpec.STUB
}
