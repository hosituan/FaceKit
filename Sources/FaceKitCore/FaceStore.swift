import CryptoKit
import Foundation

/// Persistence for enrolled identities. Implement this to store templates in your own backend.
public protocol FaceStore: Sendable {
    func loadAll() async throws -> [Identity]
    func save(_ identity: Identity) async throws
    func delete(id: String) async throws
    func deleteAll() async throws
}

public actor InMemoryFaceStore: FaceStore {
    private var identities: [String: Identity]

    public init(_ identities: [Identity] = []) {
        self.identities = Dictionary(identities.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })
    }

    public func loadAll() -> [Identity] { identities.values.sorted { $0.id < $1.id } }
    public func save(_ identity: Identity) { identities[identity.id] = identity }
    public func delete(id: String) { identities[id] = nil }
    public func deleteAll() { identities.removeAll() }
}

/// Stores all identities in a single file, optionally encrypted with AES-GCM.
///
/// Face templates are biometric data. Pass a `key` (kept in the Keychain by the app) to
/// encrypt them at rest; on iOS the file is also written with complete file protection.
public actor FileFaceStore: FaceStore {
    public let url: URL
    private let key: SymmetricKey?

    public init(url: URL, key: SymmetricKey? = nil) {
        self.url = url
        self.key = key
    }

    public func loadAll() throws -> [Identity] {
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        var data = try Data(contentsOf: url)
        if let key {
            data = try AES.GCM.open(AES.GCM.SealedBox(combined: data), using: key)
        }
        return try JSONDecoder().decode([Identity].self, from: data)
    }

    public func save(_ identity: Identity) throws {
        var all = try loadAll().filter { $0.id != identity.id }
        all.append(identity)
        try write(all)
    }

    public func delete(id: String) throws {
        try write(try loadAll().filter { $0.id != id })
    }

    public func deleteAll() throws {
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    private func write(_ identities: [Identity]) throws {
        var data = try JSONEncoder().encode(identities.sorted { $0.id < $1.id })
        if let key {
            guard let sealed = try AES.GCM.seal(data, using: key).combined else {
                throw CocoaError(.fileWriteUnknown)
            }
            data = sealed
        }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        #if os(iOS)
        try data.write(to: url, options: [.atomic, .completeFileProtection])
        #else
        try data.write(to: url, options: .atomic)
        #endif
    }
}
