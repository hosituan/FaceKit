import CoreGraphics
import CoreML
import FaceKitCore

/// FaceNet (InceptionResNetV1) converted to Core ML, fp16 weights.
///
/// Input: an aligned 160x160 RGB face crop; normalisation `(x - 128) / 128` is built into the
/// model. Output: a 128-d L2-normalised embedding. Matches the original TensorFlow graph to
/// cosine similarity > 0.9999 (see Tools/convert_model.py).
public final class FaceEmbedder: @unchecked Sendable {
    public static let inputSize = 160
    public static let embeddingDimension = 128

    private let model: MLModel
    private let imageConstraint: MLImageConstraint

    public init(configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        guard let url = Bundle.module.url(forResource: "FaceNet", withExtension: "mlmodelc") else {
            throw FaceKitError.modelNotFound
        }
        model = try MLModel(contentsOf: url, configuration: configuration)
        guard let constraint = model.modelDescription.inputDescriptionsByName["input"]?.imageConstraint else {
            throw FaceKitError.invalidModelOutput
        }
        imageConstraint = constraint
    }

    public func embedding(for face: CGImage) throws -> FaceEmbedding {
        let value = try MLFeatureValue(cgImage: face, constraint: imageConstraint, options: nil)
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input": value]))
        guard let array = output.featureValue(for: "embeddings")?.multiArrayValue,
              array.count == Self.embeddingDimension else {
            throw FaceKitError.invalidModelOutput
        }
        return FaceEmbedding((0..<array.count).map { array[$0].floatValue })
    }
}
