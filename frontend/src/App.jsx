import { useRef, useState } from "react";
import "./App.css";

import {
  detectSchema,
  generateBlockingStrategy,
  getUncertainDedupePair,
  labelDedupePair,
  prepareDedupe,
  profileDataset,
  standardizeDataset,
  trainDedupe,
  uploadDataset,
} from "./services/api";

function App() {
  const fileInputRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);

  /* ---------------- Pipeline state ---------------- */

  const [uploading, setUploading] = useState(false);
  const [profiling, setProfiling] = useState(false);
  const [detectingSchema, setDetectingSchema] = useState(false);
  const [standardizing, setStandardizing] = useState(false);

  const [uploadResult, setUploadResult] = useState(null);
  const [profileResult, setProfileResult] = useState(null);
  const [schemaResult, setSchemaResult] = useState(null);
  const [standardizationResult, setStandardizationResult] =
    useState(null);

  /* ---------------- Dedupe state ---------------- */

  const [preparingDedupe, setPreparingDedupe] = useState(false);
  const [fetchingPair, setFetchingPair] = useState(false);
  const [labelingPair, setLabelingPair] = useState(false);
  const [trainingDedupe, setTrainingDedupe] = useState(false);
  const [generatingBlocking, setGeneratingBlocking] =
    useState(false);

  const [dedupePrepared, setDedupePrepared] = useState(false);
  const [currentPair, setCurrentPair] = useState(null);

  const [matchCount, setMatchCount] = useState(0);
  const [distinctCount, setDistinctCount] = useState(0);

  const [trainingResult, setTrainingResult] = useState(null);
  const [blockingResult, setBlockingResult] = useState(null);

  const [dedupeMessage, setDedupeMessage] = useState("");

  const [error, setError] = useState("");

  /* ---------------- Helpers ---------------- */

  const processing =
    uploading ||
    profiling ||
    detectingSchema ||
    standardizing ||
    preparingDedupe ||
    fetchingPair ||
    labelingPair ||
    trainingDedupe ||
    generatingBlocking;

  function resetDedupeState() {
    setPreparingDedupe(false);
    setFetchingPair(false);
    setLabelingPair(false);
    setTrainingDedupe(false);
    setGeneratingBlocking(false);

    setDedupePrepared(false);
    setCurrentPair(null);

    setMatchCount(0);
    setDistinctCount(0);

    setTrainingResult(null);
    setBlockingResult(null);
    setDedupeMessage("");
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0];

    setError("");

    setUploadResult(null);
    setProfileResult(null);
    setSchemaResult(null);
    setStandardizationResult(null);

    resetDedupeState();

    if (!file) {
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  }

  async function handleUpload() {
    if (!selectedFile || processing) {
      return;
    }

    setUploading(true);

    setError("");
    setUploadResult(null);
    setProfileResult(null);
    setSchemaResult(null);
    setStandardizationResult(null);

    resetDedupeState();

    try {
      const uploadData = await uploadDataset(selectedFile);

      setUploadResult(uploadData);

      setUploading(false);
      setProfiling(true);

      const profileData = await profileDataset(
        uploadData.dataset_id
      );

      setProfileResult(profileData);

      setProfiling(false);
      setDetectingSchema(true);

      const schemaData = await detectSchema(
        uploadData.dataset_id
      );

      setSchemaResult(schemaData);

      setDetectingSchema(false);
      setStandardizing(true);

      const standardizationData =
        await standardizeDataset(
          uploadData.dataset_id
        );

      setStandardizationResult(
        standardizationData
      );
    } catch (err) {
      setError(
        err.message || "Dataset processing failed."
      );
    } finally {
      setUploading(false);
      setProfiling(false);
      setDetectingSchema(false);
      setStandardizing(false);
    }
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  /* ---------------- Dedupe workflow ---------------- */

  async function loadNextPair(datasetId) {
    setFetchingPair(true);
    setDedupeMessage("");

    try {
      const result = await getUncertainDedupePair(
        datasetId
      );

      const pair = result.pairs?.[0] || null;

      setCurrentPair(pair);

      if (!pair) {
        setDedupeMessage(
          "No more uncertain pairs are currently available. You can train the deduplication model using the labels collected so far."
        );
      }
    } catch (err) {
      setError(
        err.message ||
          "Failed to fetch the next uncertain pair."
      );
    } finally {
      setFetchingPair(false);
    }
  }

  async function handlePrepareDedupe() {
    if (!uploadResult?.dataset_id || processing) {
      return;
    }

    const datasetId = uploadResult.dataset_id;

    setPreparingDedupe(true);
    setError("");
    setDedupeMessage("");

    try {
      await prepareDedupe(datasetId);

      setDedupePrepared(true);
      setMatchCount(0);
      setDistinctCount(0);
      setTrainingResult(null);
      setBlockingResult(null);

      await loadNextPair(datasetId);
    } catch (err) {
      setError(
        err.message ||
          "Failed to prepare deduplication."
      );
    } finally {
      setPreparingDedupe(false);
    }
  }

  async function handleLabel(label) {
    if (
      !uploadResult?.dataset_id ||
      !currentPair ||
      labelingPair ||
      trainingDedupe
    ) {
      return;
    }

    const datasetId = uploadResult.dataset_id;
    const pairBeingLabeled = currentPair;

    setLabelingPair(true);
    setError("");

    try {
      await labelDedupePair(
        datasetId,
        pairBeingLabeled,
        label
      );

      if (label === "match") {
        setMatchCount((count) => count + 1);
      } else {
        setDistinctCount((count) => count + 1);
      }

      setCurrentPair(null);

      await loadNextPair(datasetId);
    } catch (err) {
      setError(
        err.message ||
          "Failed to save the dedupe label."
      );
    } finally {
      setLabelingPair(false);
    }
  }

  async function handleTrainDedupe() {
    if (
      !uploadResult?.dataset_id ||
      trainingDedupe ||
      labelingPair
    ) {
      return;
    }

    const datasetId = uploadResult.dataset_id;

    setTrainingDedupe(true);
    setError("");
    setDedupeMessage("");

    try {
      const result = await trainDedupe(datasetId);

      setTrainingResult(result);
      setCurrentPair(null);

      setGeneratingBlocking(true);

      try {
        const blockingData =
          await generateBlockingStrategy(
            datasetId
          );

        setBlockingResult(blockingData);
      } catch (blockingError) {
        setError(
          blockingError.message ||
            "Model trained, but blocking strategy generation failed."
        );
      } finally {
        setGeneratingBlocking(false);
      }
    } catch (err) {
      setError(
        err.message ||
          "Failed to train the deduplication model."
      );
    } finally {
      setTrainingDedupe(false);
    }
  }

  const canTrain =
    matchCount >= 3 && distinctCount >= 3;

  const hasEnoughLabelsForRecommendation =
    matchCount >= 8 && distinctCount >= 8;

  const dedupeReady =
    dedupePrepared && !trainingResult;

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <p className="eyebrow">DATAZEN</p>
          <h1>Intelligent Deduplication</h1>
        </div>

        <div className="status">
          <span className="status-dot" />
          {processing ? "Processing" : "System Ready"}
        </div>
      </header>

      <main className="main-content">
        {/* ---------------- HERO ---------------- */}

        <section className="hero">
          <p className="eyebrow">ENTITY RESOLUTION</p>

          <h2>
            Find duplicates.
            <br />
            Build trusted data.
          </h2>

          <p className="hero-text">
            Upload your dataset and let the deduplication
            pipeline identify potential duplicate entities
            across records.
          </p>
        </section>

        {/* ---------------- PIPELINE ---------------- */}

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
              <strong>Upload</strong>
              <small>Load dataset</small>
            </div>

            <div className="pipeline-line" />

            <div
              className={`pipeline-step ${
                profileResult || profiling
                  ? "active"
                  : ""
              }`}
            >
              <span>02</span>
              <strong>Profile</strong>
              <small>Understand data</small>
            </div>

            <div className="pipeline-line" />

            <div
              className={`pipeline-step ${
                schemaResult || detectingSchema
                  ? "active"
                  : ""
              }`}
            >
              <span>03</span>
              <strong>Schema</strong>
              <small>Detect semantics</small>
            </div>

            <div className="pipeline-line" />

            <div
              className={`pipeline-step ${
                standardizationResult ||
                standardizing
                  ? "active"
                  : ""
              }`}
            >
              <span>04</span>
              <strong>Standardize</strong>
              <small>Normalize fields</small>
            </div>

            <div className="pipeline-line" />

            <div
              className={`pipeline-step ${
                dedupePrepared ? "active" : ""
              }`}
            >
              <span>05</span>
              <strong>Learn</strong>
              <small>Label pairs</small>
            </div>

            <div className="pipeline-line" />

            <div
              className={`pipeline-step ${
                trainingResult ? "active" : ""
              }`}
            >
              <span>06</span>
              <strong>Train</strong>
              <small>Learn matching</small>
            </div>

            <div className="pipeline-line" />

            <div
              className={`pipeline-step ${
                blockingResult ? "active" : ""
              }`}
            >
              <span>07</span>
              <strong>Block</strong>
              <small>Reduce candidates</small>
            </div>

            <div className="pipeline-line" />

            <div className="pipeline-step">
              <span>08</span>
              <strong>Match</strong>
              <small>Find duplicates</small>
            </div>
          </div>
        </section>

        {/* ---------------- UPLOAD ---------------- */}

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
            Upload a CSV or Excel file to begin the
            deduplication process.
          </p>

          <button
            type="button"
            className="upload-button secondary-button"
            onClick={openFilePicker}
            disabled={processing}
          >
            Choose Dataset
          </button>

          {selectedFile && (
            <div className="selected-file">
              <strong>{selectedFile.name}</strong>

              <span>
                {(
                  selectedFile.size /
                  (1024 * 1024)
                ).toFixed(2)}{" "}
                MB
              </span>
            </div>
          )}

          {selectedFile && !uploadResult && (
            <button
              type="button"
              className="upload-button"
              onClick={handleUpload}
              disabled={processing}
            >
              {uploading
                ? "Uploading..."
                : "Upload & Analyze"}
            </button>
          )}

          {uploading && (
            <div className="message processing">
              <strong>
                Uploading dataset...
              </strong>

              <span>
                Saving your dataset securely before
                analysis.
              </span>
            </div>
          )}

          {profiling && (
            <div className="message processing">
              <strong>
                Profiling dataset...
              </strong>

              <span>
                Analyzing rows, columns, nulls, and
                unique values.
              </span>
            </div>
          )}

          {detectingSchema && (
            <div className="message processing">
              <strong>
                Detecting semantic fields...
              </strong>

              <span>
                Identifying identifiers, names, emails,
                and other field meanings.
              </span>
            </div>
          )}

          {standardizing && (
            <div className="message processing">
              <strong>
                Standardizing dataset...
              </strong>

              <span>
                Applying semantic-aware cleaning and
                normalization rules.
              </span>
            </div>
          )}

          {error && (
            <div className="message error">
              <strong>Processing failed.</strong>
              <span>{error}</span>
            </div>
          )}

          {uploadResult &&
            !processing &&
            !error && (
              <div className="message success">
                <strong>
                  Dataset processed successfully.
                </strong>

                <span>
                  Dataset ID:{" "}
                  {uploadResult.dataset_id}
                </span>

                <span>
                  File:{" "}
                  {uploadResult.original_filename}
                </span>

                <span>
                  Size:{" "}
                  {uploadResult.file_size_bytes.toLocaleString()}{" "}
                  bytes
                </span>
              </div>
            )}

          <small>
            Supported formats: CSV, XLSX, XLS
          </small>
        </section>

        {/* ---------------- PROFILE ---------------- */}

        {profileResult && (
          <section className="profile-card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">
                  PROFILE COMPLETE
                </p>

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
                  {profileResult.columns.map(
                    (column) => (
                      <tr key={column.name}>
                        <td>
                          <strong>
                            {column.name}
                          </strong>
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
                    )
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* ---------------- SCHEMA ---------------- */}

        {schemaResult && (
          <section className="schema-card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">
                  SCHEMA DETECTED
                </p>

                <h3>
                  Semantic Field Detection
                </h3>
              </div>

              <div className="schema-count">
                {schemaResult.fields.length} Fields
              </div>
            </div>

            <p className="schema-description">
              Each dataset column has been assigned a
              semantic meaning to guide downstream
              standardization and matching.
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
                  {schemaResult.fields.map(
                    (field) => (
                      <tr
                        key={field.column_name}
                      >
                        <td>
                          <strong>
                            {field.column_name}
                          </strong>
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
                                    field.confidence *
                                      100
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
                              (
                                evidence,
                                index
                              ) => (
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
                    )
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* ---------------- STANDARDIZATION ---------------- */}

        {standardizationResult && (
          <section className="standardization-card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">
                  STANDARDIZATION COMPLETE
                </p>

                <h3>
                  Standardized Dataset
                </h3>
              </div>

              <div className="schema-count">
                {
                  standardizationResult.column_count
                }{" "}
                Columns
              </div>
            </div>

            <p className="schema-description">
              Standardization rules were derived from
              the detected semantic field types and
              applied without modifying the original
              uploaded dataset.
            </p>

            <div className="standardization-plan">
              <h4>Applied Transformations</h4>

              <div className="standardization-fields">
                {standardizationResult.plan.fields.map(
                  (field) => (
                    <div
                      className="standardization-field"
                      key={field.column_name}
                    >
                      <div className="standardization-field-header">
                        <strong>
                          {field.column_name}
                        </strong>

                        <span className="semantic-badge">
                          {field.semantic_type}
                        </span>
                      </div>

                      <div className="transformation-list">
                        {field.transformations.map(
                          (
                            transformation,
                            index
                          ) => (
                            <span
                              key={`${field.column_name}-transformation-${index}`}
                            >
                              {transformation}
                            </span>
                          )
                        )}
                      </div>
                    </div>
                  )
                )}
              </div>
            </div>

            <div className="standardized-preview">
              <div className="preview-heading">
                <div>
                  <h4>
                    Standardized Data Preview
                  </h4>

                  <span>
                    Showing the first standardized
                    records
                  </span>
                </div>

                <div className="profile-summary">
                  <div>
                    <strong>
                      {standardizationResult.row_count.toLocaleString()}
                    </strong>

                    <span>Rows</span>
                  </div>
                </div>
              </div>

              <div className="profile-table-wrapper">
                <table className="profile-table standardized-table">
                  <thead>
                    <tr>
                      {standardizationResult.columns.map(
                        (column) => (
                          <th key={column}>
                            {column}
                          </th>
                        )
                      )}
                    </tr>
                  </thead>

                  <tbody>
                    {standardizationResult.preview.map(
                      (row, rowIndex) => (
                        <tr key={rowIndex}>
                          {standardizationResult.columns.map(
                            (column) => (
                              <td
                                key={`${rowIndex}-${column}`}
                              >
                                {row[column] ===
                                  null ||
                                row[column] ===
                                  undefined
                                  ? "—"
                                  : String(
                                      row[column]
                                    )}
                              </td>
                            )
                          )}
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}

        {/* ---------------- DEDUPE ---------------- */}

        {standardizationResult && (
          <section className="dedupe-card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">
                  DEDUPLICATION
                </p>

                <h3>
                  Learn Which Records Match
                </h3>
              </div>

              {dedupePrepared && (
                <div className="schema-count">
                  Active Learning
                </div>
              )}
            </div>

            {!dedupePrepared && (
              <>
                <p className="schema-description">
                  The system will prepare representative
                  record pairs and ask you to identify
                  whether each pair refers to the same
                  entity or two distinct entities.
                </p>

                <div className="dedupe-start">
                  <div className="dedupe-start-icon">
                    AI
                  </div>

                  <div className="dedupe-start-content">
                    <strong>
                      Start dedupe learning
                    </strong>

                    <span>
                      Your labels will be used to learn
                      matching predicates automatically.
                    </span>
                  </div>

                  <button
                    type="button"
                    className="upload-button"
                    onClick={handlePrepareDedupe}
                    disabled={processing}
                  >
                    {preparingDedupe
                      ? "Preparing..."
                      : "Start Dedupe Learning"}
                  </button>
                </div>
              </>
            )}

            {dedupePrepared &&
              !trainingResult && (
                <>
                  <div className="label-progress">
                    <div className="label-stat match-stat">
                      <span>Matches</span>
                      <strong>{matchCount}</strong>
                    </div>

                    <div className="label-stat distinct-stat">
                      <span>Distinct</span>
                      <strong>
                        {distinctCount}
                      </strong>
                    </div>

                    <div className="label-guidance">
                      <strong>
                        {canTrain
                          ? "Ready to train"
                          : "Collect more labels"}
                      </strong>

                      <span>
                        {hasEnoughLabelsForRecommendation
                          ? "Good label coverage. You can train the model now."
                          : "Recommended: collect at least 8 examples of each class for stronger learning."}
                      </span>
                    </div>
                  </div>

                  {currentPair && (
                    <div className="pair-card">
                      <div className="pair-heading">
                        <div>
                          <p className="eyebrow">
                            UNCERTAIN PAIR
                          </p>

                          <h4>
                            Are these the same
                            entity?
                          </h4>
                        </div>

                        <div className="pair-number">
                          Pair
                        </div>
                      </div>

                      <div className="pair-comparison">
                        <div className="record-card">
                          <div className="record-header">
                            <span>
                              RECORD A
                            </span>

                            <strong>
                              #
                              {
                                currentPair
                                  .record_a
                                  .record_id
                              }
                            </strong>
                          </div>

                          <div className="record-fields">
                            {Object.entries(
                              currentPair.record_a
                                .data
                            ).map(
                              ([field, value]) => (
                                <div
                                  className="record-field"
                                  key={`a-${field}`}
                                >
                                  <span>
                                    {field}
                                  </span>

                                  <strong>
                                    {value === ""
                                      ? "—"
                                      : value}
                                  </strong>
                                </div>
                              )
                            )}
                          </div>
                        </div>

                        <div className="versus">
                          VS
                        </div>

                        <div className="record-card">
                          <div className="record-header">
                            <span>
                              RECORD B
                            </span>

                            <strong>
                              #
                              {
                                currentPair
                                  .record_b
                                  .record_id
                              }
                            </strong>
                          </div>

                          <div className="record-fields">
                            {Object.entries(
                              currentPair.record_b
                                .data
                            ).map(
                              ([field, value]) => (
                                <div
                                  className="record-field"
                                  key={`b-${field}`}
                                >
                                  <span>
                                    {field}
                                  </span>

                                  <strong>
                                    {value === ""
                                      ? "—"
                                      : value}
                                  </strong>
                                </div>
                              )
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="label-actions">
                        <button
                          type="button"
                          className="label-button distinct-button"
                          onClick={() =>
                            handleLabel(
                              "distinct"
                            )
                          }
                          disabled={
                            labelingPair ||
                            fetchingPair ||
                            trainingDedupe
                          }
                        >
                          <span>×</span>
                          Distinct
                        </button>

                        <button
                          type="button"
                          className="label-button match-button"
                          onClick={() =>
                            handleLabel("match")
                          }
                          disabled={
                            labelingPair ||
                            fetchingPair ||
                            trainingDedupe
                          }
                        >
                          <span>✓</span>
                          Match
                        </button>
                      </div>
                    </div>
                  )}

                  {fetchingPair &&
                    !currentPair && (
                      <div className="message processing">
                        <strong>
                          Finding the next uncertain
                          pair...
                        </strong>

                        <span>
                          The active-learning model is
                          selecting the most informative
                          comparison.
                        </span>
                      </div>
                    )}

                  {dedupeMessage && (
                    <div className="message processing">
                      <strong>
                        Active learning paused
                      </strong>

                      <span>
                        {dedupeMessage}
                      </span>
                    </div>
                  )}

                  <div className="training-panel">
                    <div>
                      <strong>
                        Train deduplication model
                      </strong>

                      <span>
                        {canTrain
                          ? "Enough labels have been collected to train."
                          : `Need at least 3 Match and 3 Distinct labels (${matchCount}/3, ${distinctCount}/3).`}
                      </span>
                    </div>

                    <button
                      type="button"
                      className="upload-button"
                      onClick={
                        handleTrainDedupe
                      }
                      disabled={
                        !canTrain ||
                        processing
                      }
                    >
                      {trainingDedupe
                        ? "Training..."
                        : "Train Dedupe Model"}
                    </button>
                  </div>
                </>
              )}

            {trainingResult && (
              <div className="trained-section">
                <div className="trained-banner">
                  <div className="trained-icon">
                    ✓
                  </div>

                  <div>
                    <strong>
                      Dedupe model trained
                    </strong>

                    <span>
                      The model learned matching
                      patterns from your labeled
                      examples.
                    </span>
                  </div>
                </div>

                <div className="learned-patterns">
                  <div className="subsection-heading">
                    <div>
                      <p className="eyebrow">
                        LEARNED PREDICATES
                      </p>

                      <h4>
                        Matching Patterns
                      </h4>
                    </div>
                  </div>

                  {trainingResult.learned_patterns
                    ?.length > 0 ? (
                    <div className="pattern-grid">
                      {trainingResult.learned_patterns.map(
                        (pattern, index) => (
                          <div
                            className="pattern-card"
                            key={`${pattern.fields.join(
                              "-"
                            )}-${index}`}
                          >
                            <div className="pattern-card-top">
                              <span className="pattern-icon">
                                ↔
                              </span>

                              <span className="semantic-badge">
                                {
                                  pattern.pattern_type
                                }
                              </span>
                            </div>

                            <strong>
                              {pattern.fields.join(
                                " + "
                              )}
                            </strong>

                            <span>
                              {pattern.description}
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  ) : (
                    <div className="message error">
                      <strong>
                        No learned patterns returned.
                      </strong>

                      <span>
                        The model trained, but no
                        blocking pattern was produced.
                      </span>
                    </div>
                  )}
                </div>

                {generatingBlocking && (
                  <div className="message processing">
                    <strong>
                      Generating blocking strategy...
                    </strong>

                    <span>
                      Converting learned patterns into
                      conservative Splink candidate
                      blocking rules.
                    </span>
                  </div>
                )}

                {blockingResult && (
                  <div className="blocking-section">
                    <div className="subsection-heading">
                      <div>
                        <p className="eyebrow">
                          BLOCKING STRATEGY
                        </p>

                        <h4>
                          Splink Candidate Rules
                        </h4>
                      </div>

                      <div className="schema-count">
                        {blockingResult.rules.length}{" "}
                        Rule
                        {blockingResult.rules.length !==
                        1
                          ? "s"
                          : ""}
                      </div>
                    </div>

                    <p className="schema-description">
                      These rules reduce the number of
                      record pairs that need expensive
                      pairwise comparison while
                      preserving the learned matching
                      signal.
                    </p>

                    <div className="blocking-rules">
                      {blockingResult.rules.map(
                        (rule, index) => (
                          <div
                            className="blocking-rule"
                            key={`${rule.fields.join(
                              "-"
                            )}-${index}`}
                          >
                            <div className="rule-number">
                              {String(
                                index + 1
                              ).padStart(2, "0")}
                            </div>

                            <div className="rule-content">
                              <div className="rule-header">
                                <strong>
                                  Exact blocking
                                </strong>

                                <span className="semantic-badge">
                                  {rule.fields.join(
                                    " + "
                                  )}
                                </span>
                              </div>

                              <code>
                                {rule.sql_condition}
                              </code>
                            </div>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}

                {blockingResult && (
                  <div className="ready-banner">
                    <div className="ready-icon">
                      →
                    </div>

                    <div>
                      <strong>
                        Ready for candidate generation
                      </strong>

                      <span>
                        The learned blocking strategy is
                        ready for the next pipeline stage:
                        candidate pair generation.
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;