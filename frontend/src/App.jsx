import { useRef, useState } from "react";
import "./App.css";
import {
  detectSchema,
  profileDataset,
  uploadDataset,
} from "./services/api";

function App() {
  const fileInputRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [profiling, setProfiling] = useState(false);
  const [detectingSchema, setDetectingSchema] = useState(false);

  const [uploadResult, setUploadResult] = useState(null);
  const [profileResult, setProfileResult] = useState(null);
  const [schemaResult, setSchemaResult] = useState(null);
  const [error, setError] = useState("");

  function handleFileChange(event) {
    const file = event.target.files?.[0];

    setError("");
    setUploadResult(null);
    setProfileResult(null);
    setSchemaResult(null);

    if (!file) {
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  }

  async function handleUpload() {
    if (
      !selectedFile ||
      uploading ||
      profiling ||
      detectingSchema
    ) {
      return;
    }

    setUploading(true);
    setProfiling(false);
    setDetectingSchema(false);

    setError("");
    setUploadResult(null);
    setProfileResult(null);
    setSchemaResult(null);

    try {
      // Step 1: Upload dataset
      const uploadData = await uploadDataset(selectedFile);
      setUploadResult(uploadData);

      // Step 2: Profile dataset
      setUploading(false);
      setProfiling(true);

      const profileData = await profileDataset(
        uploadData.dataset_id
      );
      setProfileResult(profileData);

      // Step 3: Detect semantic schema
      setProfiling(false);
      setDetectingSchema(true);

      const schemaData = await detectSchema(
        uploadData.dataset_id
      );
      setSchemaResult(schemaData);
    } catch (err) {
      setError(err.message || "Dataset processing failed.");
    } finally {
      setUploading(false);
      setProfiling(false);
      setDetectingSchema(false);
    }
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <p className="eyebrow">DATAZEN</p>
          <h1>Intelligent Deduplication</h1>
        </div>

        <div className="status">
          <span className="status-dot" />
          System Ready
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <p className="eyebrow">ENTITY RESOLUTION</p>

          <h2>
            Find duplicates.
            <br />
            Build trusted data.
          </h2>

          <p className="hero-text">
            Upload your dataset and let the deduplication pipeline
            identify potential duplicate entities across records.
          </p>
        </section>

        <section className="pipeline-card">
          <div className="section-heading">
            <div>
              <p className="eyebrow">PIPELINE</p>
              <h3>Deduplication Workflow</h3>
            </div>
          </div>

          <div className="pipeline">
            <div
              className={`pipeline-step ${
                uploadResult ? "active" : ""
              }`}
            >
              <span>01</span>
              <strong>Upload Data</strong>
              <small>Start with your dataset</small>
            </div>

            <div className="pipeline-line" />

            <div
              className={`pipeline-step ${
                profileResult || profiling ? "active" : ""
              }`}
            >
              <span>02</span>
              <strong>Profile</strong>
              <small>Understand the data</small>
            </div>

            <div className="pipeline-line" />

            <div
              className={`pipeline-step ${
                schemaResult || detectingSchema ? "active" : ""
              }`}
            >
              <span>03</span>
              <strong>Schema</strong>
              <small>Detect semantic fields</small>
            </div>

            <div className="pipeline-line" />

            <div className="pipeline-step">
              <span>04</span>
              <strong>Standardize</strong>
              <small>Clean and normalize</small>
            </div>

            <div className="pipeline-line" />

            <div className="pipeline-step">
              <span>05</span>
              <strong>Match</strong>
              <small>Find duplicate records</small>
            </div>
          </div>
        </section>

        <section className="upload-card">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileChange}
            hidden
          />

          <div className="upload-icon">↑</div>

          <h3>Upload your dataset</h3>

          <p>
            Upload a CSV or Excel file to begin the deduplication
            process.
          </p>

          <button
            type="button"
            className="upload-button secondary-button"
            onClick={openFilePicker}
            disabled={
              uploading ||
              profiling ||
              detectingSchema
            }
          >
            Choose Dataset
          </button>

          {selectedFile && (
            <div className="selected-file">
              <strong>{selectedFile.name}</strong>

              <span>
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </span>
            </div>
          )}

          {selectedFile && !uploadResult && (
            <button
              type="button"
              className="upload-button"
              onClick={handleUpload}
              disabled={
                uploading ||
                profiling ||
                detectingSchema
              }
            >
              {uploading
                ? "Uploading..."
                : "Upload & Analyze"}
            </button>
          )}

          {uploading && (
            <div className="message processing">
              <strong>Uploading dataset...</strong>

              <span>
                Saving your dataset securely before analysis.
              </span>
            </div>
          )}

          {profiling && (
            <div className="message processing">
              <strong>Profiling dataset...</strong>

              <span>
                Analyzing rows, columns, nulls, and unique values.
              </span>
            </div>
          )}

          {detectingSchema && (
            <div className="message processing">
              <strong>Detecting semantic fields...</strong>

              <span>
                Identifying identifiers, names, emails, and other
                field meanings.
              </span>
            </div>
          )}

          {error && (
            <div className="message error">
              <strong>Processing failed.</strong>
              <span>{error}</span>
            </div>
          )}

          {uploadResult && !profiling && !detectingSchema && (
            <div className="message success">
              <strong>Dataset uploaded successfully.</strong>

              <span>
                Dataset ID: {uploadResult.dataset_id}
              </span>

              <span>
                File: {uploadResult.original_filename}
              </span>

              <span>
                Size:{" "}
                {uploadResult.file_size_bytes.toLocaleString()} bytes
              </span>
            </div>
          )}

          <small>Supported formats: CSV, XLSX, XLS</small>
        </section>

        {profileResult && (
          <section className="profile-card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">PROFILE COMPLETE</p>
                <h3>Dataset Profile</h3>
              </div>

              <div className="profile-summary">
                <div>
                  <strong>
                    {profileResult.row_count.toLocaleString()}
                  </strong>
                  <span>Rows</span>
                </div>

                <div>
                  <strong>
                    {profileResult.column_count}
                  </strong>
                  <span>Columns</span>
                </div>
              </div>
            </div>

            <div className="profile-table-wrapper">
              <table className="profile-table">
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Nulls</th>
                    <th>Unique</th>
                    <th>Sample Values</th>
                  </tr>
                </thead>

                <tbody>
                  {profileResult.columns.map((column) => (
                    <tr key={column.name}>
                      <td>
                        <strong>{column.name}</strong>
                      </td>

                      <td>
                        <span className="type-badge">
                          {column.dtype}
                        </span>
                      </td>

                      <td>
                        {column.null_count.toLocaleString()}{" "}
                        <small>
                          ({column.null_percentage}%)
                        </small>
                      </td>

                      <td>
                        {column.unique_count.toLocaleString()}{" "}
                        <small>
                          ({column.unique_percentage}%)
                        </small>
                      </td>

                      <td>
                        <div className="sample-values">
                          {column.sample_values.map(
                            (value, index) => (
                              <span
                                key={`${column.name}-${index}`}
                              >
                                {value}
                              </span>
                            )
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {schemaResult && (
          <section className="schema-card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">SCHEMA DETECTED</p>
                <h3>Semantic Field Detection</h3>
              </div>

              <div className="schema-count">
                {schemaResult.fields.length} Fields
              </div>
            </div>

            <p className="schema-description">
              Each dataset column has been assigned a semantic
              meaning to guide the downstream standardization and
              matching stages.
            </p>

            <div className="schema-table-wrapper">
              <table className="schema-table">
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Semantic Type</th>
                    <th>Confidence</th>
                    <th>Evidence</th>
                  </tr>
                </thead>

                <tbody>
                  {schemaResult.fields.map((field) => (
                    <tr key={field.column_name}>
                      <td>
                        <strong>{field.column_name}</strong>
                      </td>

                      <td>
                        <span className="semantic-badge">
                          {field.semantic_type}
                        </span>
                      </td>

                      <td>
                        <div className="confidence">
                          <div className="confidence-bar">
                            <span
                              style={{
                                width: `${Math.round(
                                  field.confidence * 100
                                )}%`,
                              }}
                            />
                          </div>

                          <strong>
                            {(
                              field.confidence * 100
                            ).toFixed(0)}
                            %
                          </strong>
                        </div>
                      </td>

                      <td>
                        <div className="evidence-list">
                          {field.evidence.map(
                            (evidence, index) => (
                              <span
                                key={`${field.column_name}-evidence-${index}`}
                              >
                                {evidence}
                              </span>
                            )
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;