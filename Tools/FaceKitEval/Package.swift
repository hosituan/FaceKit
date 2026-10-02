// swift-tools-version:5.9
import PackageDescription

// Benchmark tool for FaceKit: verification accuracy and threshold calibration on LFW.
// Kept out of the main package so consumers never resolve it.
let package = Package(
    name: "FaceKitEval",
    platforms: [.macOS(.v12)],
    dependencies: [.package(path: "../..")],
    targets: [
        .executableTarget(name: "FaceKitEval", dependencies: [.product(name: "FaceKit", package: "FaceKit")]),
    ]
)
