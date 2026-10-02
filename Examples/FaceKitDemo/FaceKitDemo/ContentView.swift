import AVFoundation
import FaceKit
import SwiftUI

struct ContentView: View {
    @StateObject private var model = FaceDemoModel()
    @State private var name = ""
    @State private var showingPeople = false

    var body: some View {
        ZStack {
            CameraPreview(session: model.session)
                .ignoresSafeArea()

            GeometryReader { geometry in
                ForEach(Array(model.recognitions.enumerated()), id: \.offset) { _, recognition in
                    FaceBox(recognition: recognition,
                            rect: Self.screenRect(of: recognition.face, frame: model.frameSize, view: geometry.size))
                }
            }
            .ignoresSafeArea()

            VStack {
                if !model.status.isEmpty {
                    Text(model.status)
                        .font(.callout)
                        .padding(8)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                }
                Spacer()
                controls
            }
            .padding()
        }
        .task { await model.start() }
        .sheet(isPresented: $showingPeople) { PeopleView(model: model) }
    }

    private var controls: some View {
        HStack {
            TextField("Name", text: $name)
                .textFieldStyle(.roundedBorder)
                .disableAutocorrection(true)
            Button("Enroll") {
                let enrolled = name.trimmingCharacters(in: .whitespaces)
                Task { await model.enroll(name: enrolled) }
                name = ""
            }
            .buttonStyle(.borderedProminent)
            .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty || model.isEnrolling)
            Button {
                showingPeople = true
            } label: {
                Label("\(model.people.count)", systemImage: "person.2")
            }
            .buttonStyle(.bordered)
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    /// Maps a face in the upright camera frame onto the aspect-filled preview.
    static func screenRect(of face: DetectedFace, frame: CGSize, view: CGSize) -> CGRect {
        guard frame.width > 0, frame.height > 0 else { return .zero }
        let scale = max(view.width / frame.width, view.height / frame.height)
        let offsetX = (view.width - frame.width * scale) / 2
        let offsetY = (view.height - frame.height * scale) / 2
        let box = face.boundingBoxTopLeft
        return CGRect(x: box.minX * frame.width * scale + offsetX,
                      y: box.minY * frame.height * scale + offsetY,
                      width: box.width * frame.width * scale,
                      height: box.height * frame.height * scale)
    }
}

private struct FaceBox: View {
    let recognition: Recognition
    let rect: CGRect

    var body: some View {
        let match = recognition.match
        let color: Color = match.decision == .confident ? .green : match.decision == .candidate ? .yellow : .red
        let name = match.identityID.map { match.decision == .candidate ? "\($0)?" : $0 } ?? "Unknown"
        RoundedRectangle(cornerRadius: 8)
            .stroke(color, lineWidth: 3)
            .frame(width: rect.width, height: rect.height)
            .position(x: rect.midX, y: rect.midY)
        Text(match.distance.isFinite ? "\(name) · \(String(format: "%.2f", match.distance))" : name)
            .font(.headline)
            .padding(.horizontal, 6)
            .background(color)
            .foregroundColor(.black)
            .position(x: rect.midX, y: max(rect.minY - 14, 14))
    }
}

private struct PeopleView: View {
    @ObservedObject var model: FaceDemoModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            List {
                if model.people.isEmpty {
                    Text("Nobody enrolled yet. Type a name and tap Enroll while looking at the camera.")
                        .foregroundColor(.secondary)
                }
                ForEach(model.people, id: \.self) { Text($0) }
                    .onDelete { offsets in
                        let names = offsets.map { model.people[$0] }
                        Task { for name in names { await model.remove(name) } }
                    }
            }
            .navigationTitle("Enrolled people")
            .toolbar { Button("Done") { dismiss() } }
        }
    }
}

/// AVCaptureVideoPreviewLayer as a SwiftUI view.
private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}
