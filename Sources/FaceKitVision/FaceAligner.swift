import CoreImage

/// Turns a detected face into the square crop the embedding model expects:
/// levelled eyes (roll removed), the face box plus a margin, scaled to `outputSize`.
public struct FaceAligner: Sendable {
    public var outputSize: Int
    /// Extra context added on each side, as a fraction of the larger face-box side.
    public var margin: CGFloat
    public var correctsRoll: Bool

    public init(outputSize: Int = 160, margin: CGFloat = 0.2, correctsRoll: Bool = true) {
        self.outputSize = outputSize
        self.margin = margin
        self.correctsRoll = correctsRoll
    }

    /// - Parameter image: the same upright image the face was detected in.
    public func align(_ face: DetectedFace, in image: CIImage) -> CIImage {
        let out = CGFloat(outputSize)
        return image.clampedToExtent()
            .transformed(by: alignmentTransform(for: face, extent: image.extent))
            .cropped(to: CGRect(x: 0, y: 0, width: out, height: out))
    }

    /// Maps image coordinates to crop coordinates: face-box centre to the crop centre,
    /// eyes levelled, face box plus margin scaled to `outputSize`.
    func alignmentTransform(for face: DetectedFace, extent: CGRect) -> CGAffineTransform {
        let box = CGRect(x: extent.minX + face.boundingBox.minX * extent.width,
                         y: extent.minY + face.boundingBox.minY * extent.height,
                         width: face.boundingBox.width * extent.width,
                         height: face.boundingBox.height * extent.height)
        let side = max(box.width, box.height) * (1 + 2 * margin)
        let out = CGFloat(outputSize)
        return CGAffineTransform(translationX: out / 2, y: out / 2)
            .scaledBy(x: out / side, y: out / side)
            .rotated(by: -(correctsRoll ? rollAngle(of: face, extent: extent) : 0))
            .translatedBy(x: -box.midX, y: -box.midY)
    }

    /// Variants of an aligned crop used to enrich enrollment: mirror and small rotations.
    public func augmentations(of aligned: CIImage, rotation: CGFloat = .pi / 20) -> [CIImage] {
        let out = CGFloat(outputSize)
        let rect = CGRect(x: 0, y: 0, width: out, height: out)
        let mirrored = aligned
            .transformed(by: CGAffineTransform(translationX: out, y: 0).scaledBy(x: -1, y: 1))
            .cropped(to: rect)
        let rotated = [rotation, -rotation].map { angle in
            aligned.clampedToExtent()
                .transformed(by: CGAffineTransform(translationX: out / 2, y: out / 2)
                    .rotated(by: angle)
                    .translatedBy(x: -out / 2, y: -out / 2))
                .cropped(to: rect)
        }
        return [mirrored] + rotated
    }

    private func rollAngle(of face: DetectedFace, extent: CGRect) -> CGFloat {
        if let a = face.leftEye, let b = face.rightEye {
            let (left, right) = a.x <= b.x ? (a, b) : (b, a)
            let dx = (right.x - left.x) * extent.width
            let dy = (right.y - left.y) * extent.height
            if dx != 0 || dy != 0 { return atan2(dy, dx) }
        }
        // Without eye landmarks, leave the crop unrotated rather than guess.
        return 0
    }
}

/// Renders aligned crops for the embedding model. Reuse one instance; `CIContext` is expensive.
public final class FaceRenderer: @unchecked Sendable {
    private let context: CIContext

    public init(context: CIContext = CIContext()) {
        self.context = context
    }

    public func cgImage(from image: CIImage) -> CGImage? {
        context.createCGImage(image, from: image.extent, format: .RGBA8,
                              colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
    }
}
