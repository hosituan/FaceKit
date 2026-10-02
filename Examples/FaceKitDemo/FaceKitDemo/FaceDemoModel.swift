import AVFoundation
import CoreImage
import FaceKit
import SwiftUI

/// Camera + FaceKit: recognises faces in the live front-camera feed and enrolls new people.
@MainActor
final class FaceDemoModel: NSObject, ObservableObject {
    @Published private(set) var recognitions: [Recognition] = []
    /// Size of the upright camera frame the recognitions refer to.
    @Published private(set) var frameSize: CGSize = .zero
    @Published private(set) var people: [String] = []
    @Published private(set) var status = "Loading model…"
    @Published private(set) var isEnrolling = false

    let session = AVCaptureSession()
    private let videoQueue = DispatchQueue(label: "FaceKitDemo.camera")
    private let context = CIContext()
    private var recognizer: FaceRecognizer?
    private var isRecognizing = false
    private var latestFrame: CVPixelBuffer?

    func start() async {
        do {
            // A real app should pass `key:` (kept in the Keychain) to encrypt the templates.
            recognizer = try await FaceRecognizer(store: FileFaceStore(url: Self.storeURL))
            await refreshPeople()
            status = ""
        } catch {
            status = "Could not load FaceKit: \(error.localizedDescription)"
            return
        }
        guard await AVCaptureDevice.requestAccess(for: .video) else {
            status = "Camera access denied. Enable it in Settings."
            return
        }
        configureSession()
    }

    /// Captures ~3 s of frames while the person turns their head slightly, then enrolls them.
    func enroll(name: String) async {
        guard let recognizer, !name.isEmpty, !isEnrolling else { return }
        isEnrolling = true
        defer { isEnrolling = false }

        var images: [CGImage] = []
        for index in 1...10 {
            status = "Capturing \(index)/10 – turn your head slowly"
            if let frame = latestFrame {
                let image = CIImage(cvPixelBuffer: frame)
                if let cgImage = context.createCGImage(image, from: image.extent) { images.append(cgImage) }
            }
            try? await Task.sleep(nanoseconds: 300_000_000)
        }
        status = "Enrolling \(name)…"
        do {
            let identity = try await recognizer.enroll(id: name, images: images)
            status = "Enrolled \(name) with \(identity.templates.count) templates"
        } catch FaceKitError.noUsableFaces {
            status = "No face found. Look at the camera and try again."
        } catch {
            status = "Enrollment failed: \(error.localizedDescription)"
        }
        await refreshPeople()
    }

    func remove(_ name: String) async {
        try? await recognizer?.remove(id: name)
        await refreshPeople()
    }

    private func refreshPeople() async {
        people = (await recognizer?.identities.map(\.id) ?? []).sorted()
    }

    private func configureSession() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device) else {
            status = "No front camera. Run the demo on an iPhone or iPad."
            return
        }
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)

        session.beginConfiguration()
        session.sessionPreset = .high
        if session.canAddInput(input) { session.addInput(input) }
        if session.canAddOutput(output) { session.addOutput(output) }
        // Ask for upright, mirrored frames that match the preview, so FaceKit needs no orientation
        // and the boxes map straight onto the screen.
        if let connection = output.connection(with: .video) {
            if #available(iOS 17.0, *) {
                if connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }
            } else {
                connection.videoOrientation = .portrait
            }
            if connection.isVideoMirroringSupported { connection.isVideoMirrored = true }
        }
        session.commitConfiguration()
        videoQueue.async { [session] in session.startRunning() }
    }

    private func process(_ frame: CVPixelBuffer) {
        latestFrame = frame
        // Recognise one frame at a time; frames arriving meanwhile are skipped.
        guard let recognizer, !isRecognizing else { return }
        isRecognizing = true
        Task {
            defer { isRecognizing = false }
            let results = (try? await recognizer.identify(in: frame)) ?? []
            frameSize = CGSize(width: CVPixelBufferGetWidth(frame), height: CVPixelBufferGetHeight(frame))
            recognitions = results
        }
    }

    private static var storeURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FaceKitDemo/identities.json")
    }
}

extension FaceDemoModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let frame = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        Task { @MainActor in self.process(frame) }
    }
}
