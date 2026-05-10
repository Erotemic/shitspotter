package io.kitware.shitspotter.core

/**
 * Placeholder implementation marker for a future LiteRT (TensorFlow Lite)
 * backend. The actual `tflite` runtime is platform-specific (the Android
 * `org.tensorflow:tensorflow-lite` AAR vs the JVM-side `tflite-runtime`),
 * so the *real* `TfliteBackend` will live in `androidMain/` and
 * `desktopMain/` like the ONNX backends do. This stub exists so that
 * future agents wiring up a TFLite path have an obvious "is the
 * model registered?" smoke test, and so that GOAL.md §"LiteRT /
 * TensorFlow Lite" is reachable from the in-tree code without forcing
 * a refactor of the ModelRegistry shape.
 *
 * If you flip a model spec to `format = ModelFormat.TFLITE` and run the
 * app today, the existing ONNX backend factories will refuse to load
 * it. That's the correct failure mode for the current scaffold.
 */
class TfliteBackendStub(
    override val spec: ModelSpec,
) : DetectorBackend {
    init {
        require(spec.format == ModelFormat.TFLITE) {
            "TfliteBackendStub requires ModelSpec.format = TFLITE, got ${spec.format}"
        }
    }

    override val backendName: String = "tflite-stub-not-implemented"
    override val delegate: String? = null

    override fun warmup(): Unit = throw NotImplementedError(
        "TFLite backend is not implemented yet — see GOAL.md §LiteRT and " +
            "tpl/shitspotter-phone-app/docs/003_known_limitations.md item #11.",
    )

    override fun analyze(frame: FrameSource): InferenceResult = throw NotImplementedError(
        "TFLite analyze() is not implemented yet — wire " +
            "androidMain/.../TfliteAndroidBackend (LiteRT AAR + GPU delegate) " +
            "and desktopMain/.../TfliteJvmBackend before flipping a ModelSpec " +
            "to ModelFormat.TFLITE.",
    )

    override fun close() = Unit
}
