import Foundation

public enum FaceKitError: Error, Equatable, Sendable {
    /// The bundled Core ML model could not be found.
    case modelNotFound
    /// The model produced an output in an unexpected shape.
    case invalidModelOutput
    /// Enrollment found no usable face in any of the supplied images.
    case noUsableFaces
}
