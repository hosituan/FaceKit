// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "FaceKit",
    platforms: [.iOS(.v15), .macOS(.v12)],
    products: [
        .library(name: "FaceKit", targets: ["FaceKit"]),
        .library(name: "FaceKitCore", targets: ["FaceKitCore"]),
    ],
    targets: [
        // Pure Swift: embeddings, matching, clustering, consensus, storage.
        .target(name: "FaceKitCore"),
        // Face detection and alignment with Vision / Core Image.
        .target(name: "FaceKitVision"),
        // FaceNet embedding model (Core ML, fp16, precompiled).
        .target(
            name: "FaceKitCoreML",
            dependencies: ["FaceKitCore"],
            resources: [.copy("Resources/FaceNet.mlmodelc")]
        ),
        // Umbrella: FaceRecognizer, re-exports the modules above.
        .target(
            name: "FaceKit",
            dependencies: ["FaceKitCore", "FaceKitVision", "FaceKitCoreML"]
        ),
        .testTarget(name: "FaceKitCoreTests", dependencies: ["FaceKitCore"]),
        .testTarget(
            name: "FaceKitCoreMLTests",
            dependencies: ["FaceKit", "FaceKitVision"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
