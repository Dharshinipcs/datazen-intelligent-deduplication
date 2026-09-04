import { useRef, useState } from "react";
import "./App.css";

import {
  buildEntityClusters,
  detectSchema,
  generateBlockingStrategy,
  generateCandidatePairs,
  getHumanReviewQueue,
  getMatchDecisions,
  getUncertainDedupePair,
  labelDedupePair,
  prepareDedupe,
  profileDataset,
  runSplinkMatching,
  standardizeDataset,
  submitHumanReview,
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

  /* ---------------- Matching pipeline state ---------------- */

  const [generatingCandidates, setGeneratingCandidates] =
    useState(false);
  const [candidateResult, setCandidateResult] = useState(null);

  const [matchingSplink, setMatchingSplink] = useState(false);
  const [matchingResult, setMatchingResult] = useState(null);

  const [classifyingMatches, setClassifyingMatches] =
    useState(false);
  const [decisionResult, setDecisionResult] = useState(null);

  /* ---------------- Human Review state ---------------- */

  const [loadingHumanReview, setLoadingHumanReview] =
    useState(false);
  const [submittingHumanReview, setSubmittingHumanReview] =
    useState(false);

  const [humanReviewResult, setHumanReviewResult] =
    useState(null);
  const [currentReviewItem, setCurrentReviewItem] =
    useState(null);

  const [reviewedMatchCount, setReviewedMatchCount] =
    useState(0);
  const [reviewedNonMatchCount, setReviewedNonMatchCount] =
    useState(0);

  const [humanReviewMessage, setHumanReviewMessage] =
    useState("");

  /* ---------------- Entity Clustering state ---------------- */

  const [clusteringEntities, setClusteringEntities] =
    useState(false);
  const [clusteringResult, setClusteringResult] =
    useState(null);
  const [clusteringMessage, setClusteringMessage] =
    useState("");

  const [error, setError] = useState("");

  /* ---------------- Processing state ---------------- */

  const processing =
    uploading ||
    profiling ||
    detectingSchema ||
    standardizing ||
    preparingDedupe ||
    fetchingPair ||
    labelingPair ||
    trainingDedupe ||
    generatingBlocking ||
    generatingCandidates ||
    matchingSplink ||
    classifyingMatches ||
    loadingHumanReview ||
    submittingHumanReview ||
    clusteringEntities;

  /* ---------------- Helpers ---------------- */

  function resetDedupeState() {
    setPreparingDedupe(false);
    setFetchingPair(false);
    setLabelingPair(false);
    setTrainingDedupe(false);
    setGeneratingBlocking(false);

    setGeneratingCandidates(false);
    setMatchingSplink(false);
    setClassifyingMatches(false);
    setLoadingHumanReview(false);
    setSubmittingHumanReview(false);

    setDedupePrepared(false);
    setCurrentPair(null);

    setMatchCount(0);
    setDistinctCount(0);

    setTrainingResult(null);
    setBlockingResult(null);

    setCandidateResult(null);
    setMatchingResult(null);
    setDecisionResult(null);

    setHumanReviewResult(null);
    setCurrentReviewItem(null);

    setReviewedMatchCount(0);
    setReviewedNonMatchCount(0);

    setDedupeMessage("");
    setHumanReviewMessage("");

    setClusteringEntities(false);
    setClusteringResult(null);
    setClusteringMessage("");
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

      setCandidateResult(null);
      setMatchingResult(null);
      setDecisionResult(null);
      setHumanReviewResult(null);
      setCurrentReviewItem(null);

      setClusteringResult(null);
      setClusteringMessage("");

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

  /* ---------------- Post-training matching pipeline ---------------- */

  async function runPostBlockingPipeline(datasetId) {
    setError("");

    let candidateData;
    let matchingData;
    let decisionData;
    let reviewData;

    /* -------- Candidate generation -------- */

    setGeneratingCandidates(true);

    try {
      candidateData =
        await generateCandidatePairs(datasetId);

      setCandidateResult(candidateData);
    } catch (err) {
      setError(
        err.message ||
          "Candidate generation failed."
      );
      return;
    } finally {
      setGeneratingCandidates(false);
    }

    /* -------- Splink matching -------- */

    setMatchingSplink(true);

    try {
      matchingData =
        await runSplinkMatching(datasetId);

      setMatchingResult(matchingData);
    } catch (err) {
      setError(
        err.message ||
          "Splink matching failed."
      );
      return;
    } finally {
      setMatchingSplink(false);
    }

    /* -------- Match decisions -------- */

    setClassifyingMatches(true);

    try {
      decisionData =
        await getMatchDecisions(datasetId);

      setDecisionResult(decisionData);
    } catch (err) {
      setError(
        err.message ||
          "Match decision classification failed."
      );
      return;
    } finally {
      setClassifyingMatches(false);
    }

    /* -------- Human Review queue -------- */

    setLoadingHumanReview(true);

    try {
      reviewData =
        await getHumanReviewQueue(datasetId);

      setHumanReviewResult(reviewData);
      setCurrentReviewItem(
        reviewData.items?.[0] || null
      );
    } catch (err) {
      setError(
        err.message ||
          "Failed to load the Human Review queue."
      );
    } finally {
      setLoadingHumanReview(false);
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

        await runPostBlockingPipeline(
          datasetId
        );
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

  /* ---------------- Human Review workflow ---------------- */

  async function handleHumanReview(decision) {
    if (
      !uploadResult?.dataset_id ||
      !currentReviewItem ||
      submittingHumanReview
    ) {
      return;
    }

    const datasetId = uploadResult.dataset_id;
    const item = currentReviewItem;

    setSubmittingHumanReview(true);
    setError("");
    setHumanReviewMessage("");

    try {
      await submitHumanReview(
        datasetId,
        item.record_a.record_id,
        item.record_b.record_id,
        decision
      );

      if (decision === "match") {
        setReviewedMatchCount(
          (count) => count + 1
        );
      } else {
        setReviewedNonMatchCount(
          (count) => count + 1
        );
      }

      setHumanReviewResult((previous) => {
        if (!previous) {
          return previous;
        }

        const remainingItems =
          previous.items.filter(
            (reviewItem) =>
              !(
                reviewItem.record_a.record_id ===
                  item.record_a.record_id &&
                reviewItem.record_b.record_id ===
                  item.record_b.record_id
              )
          );

        return {
          ...previous,
          items: remainingItems,
          review_count: remainingItems.length,
        };
      });

      setCurrentReviewItem(null);

      setHumanReviewMessage(
        decision === "match"
          ? "Review saved as Match."
          : "Review saved as Non-match."
      );

      setHumanReviewResult((previous) => {
        if (!previous) {
          return previous;
        }

        const nextItem =
          previous.items?.[0] || null;

        setCurrentReviewItem(nextItem);

        return previous;
      });
    } catch (err) {
      setError(
        err.message ||
          "Failed to save the Human Review decision."
      );
    } finally {
      setSubmittingHumanReview(false);
    }
  }

  /* ---------------- Entity Clustering workflow ---------------- */

  async function handleEntityClustering() {
    if (
      !uploadResult?.dataset_id ||
      clusteringEntities ||
      processing
    ) {
      return;
    }

    const datasetId = uploadResult.dataset_id;

    setClusteringEntities(true);
    setError("");
    setClusteringMessage("");
    setClusteringResult(null);

    try {
      const result =
        await buildEntityClusters(datasetId);

      setClusteringResult(result);

      setClusteringMessage(
        "Entity clustering completed successfully."
      );
    } catch (err) {
      setError(
        err.message ||
          "Failed to build entity clusters."
      );
    } finally {
      setClusteringEntities(false);
    }
  }

  const canTrain =
    matchCount >= 3 && distinctCount >= 3;

  const hasEnoughLabelsForRecommendation =
    matchCount >= 8 && distinctCount >= 8;

  const reviewCount =
    humanReviewResult?.review_count ?? 0;

  const humanReviewComplete =
    Boolean(humanReviewResult) &&
    reviewCount === 0;

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

            <div
              className={`pipeline-step ${
                candidateResult ||
                generatingCandidates
                  ? "active"
                  : ""
              }`}
            >
              <span>08</span>
              <strong>Candidates</strong>
              <small>Generate pairs</small>
            </div>

            <div className="pipeline-line" />

            <div
              className={`pipeline-step ${
                matchingResult ||
                matchingSplink
                  ? "active"
                  : ""
              }`}
            >
              <span>09</span>
              <strong>Match</strong>
              <small>Run Splink</small>
            </div>

            <div className="pipeline-line" />

            <div
              className={`pipeline-step ${
                decisionResult ||
                classifyingMatches
                  ? "active"
                  : ""
              }`}
            >
              <span>10</span>
              <strong>Decide</strong>
              <small>Classify pairs</small>
            </div>

            <div className="pipeline-line" />

            <div
              className={`pipeline-step ${
                humanReviewResult ||
                loadingHumanReview
                  ? "active"
                  : ""
              }`}
            >
              <span>11</span>
              <strong>Review</strong>
              <small>Human validation</small>
            </div>

            <div className="pipeline-line" />

            <div
              className={`pipeline-step ${
                clusteringResult ||
                clusteringEntities
                  ? "active"
                  : ""
              }`}
            >
              <span>12</span>
              <strong>Cluster</strong>
              <small>Group entities</small>
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

            {/* ---------------- TRAINED + MATCHING ---------------- */}

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

                {/* -------- Learned patterns -------- */}

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

                {/* -------- Blocking -------- */}

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

                {/* -------- Candidate generation -------- */}

                {generatingCandidates && (
                  <div className="message processing">
                    <strong>
                      Generating candidate pairs...
                    </strong>

                    <span>
                      Applying the learned blocking
                      rules to reduce the comparison
                      search space.
                    </span>
                  </div>
                )}

                {candidateResult && (
                  <div className="pipeline-result-card">
                    <div className="subsection-heading">
                      <div>
                        <p className="eyebrow">
                          CANDIDATE GENERATION
                        </p>

                        <h4>
                          Candidate Pairs Generated
                        </h4>
                      </div>

                      <div className="schema-count">
                        {candidateResult.candidate_pair_count.toLocaleString()}{" "}
                        Pairs
                      </div>
                    </div>

                    <div className="result-stat-grid">
                      <div className="result-stat">
                        <span>Candidate pairs</span>
                        <strong>
                          {candidateResult.candidate_pair_count.toLocaleString()}
                        </strong>
                      </div>

                      <div className="result-stat">
                        <span>Blocking rules</span>
                        <strong>
                          {candidateResult.blocking_rule_count}
                        </strong>
                      </div>

                      <div className="result-stat">
                        <span>Status</span>
                        <strong>
                          {candidateResult.status}
                        </strong>
                      </div>
                    </div>
                  </div>
                )}

                {/* -------- Splink matching -------- */}

                {matchingSplink && (
                  <div className="message processing">
                    <strong>
                      Running Splink matching...
                    </strong>

                    <span>
                      Comparing candidate records and
                      estimating match probabilities.
                    </span>
                  </div>
                )}

                {matchingResult && (
                  <div className="pipeline-result-card">
                    <div className="subsection-heading">
                      <div>
                        <p className="eyebrow">
                          SPLINK MATCHING
                        </p>

                        <h4>
                          Probabilistic Matching Complete
                        </h4>
                      </div>

                      <div className="schema-count">
                        {matchingResult.match_pair_count.toLocaleString()}{" "}
                        Matches
                      </div>
                    </div>

                    <div className="result-stat-grid">
                      <div className="result-stat">
                        <span>Pairs evaluated</span>
                        <strong>
                          {matchingResult.match_pair_count.toLocaleString()}
                        </strong>
                      </div>

                      <div className="result-stat">
                        <span>Blocking rules</span>
                        <strong>
                          {matchingResult.blocking_rule_count}
                        </strong>
                      </div>

                      <div className="result-stat">
                        <span>Status</span>
                        <strong>
                          {matchingResult.status}
                        </strong>
                      </div>
                    </div>
                  </div>
                )}

                {/* -------- Match decisions -------- */}

                {classifyingMatches && (
                  <div className="message processing">
                    <strong>
                      Classifying match probabilities...
                    </strong>

                    <span>
                      Separating automatic matches,
                      possible matches, and non-matches.
                    </span>
                  </div>
                )}

                {decisionResult && (
                  <div className="pipeline-result-card">
                    <div className="subsection-heading">
                      <div>
                        <p className="eyebrow">
                          MATCH DECISIONS
                        </p>

                        <h4>
                          Resolution Summary
                        </h4>
                      </div>

                      <div className="schema-count">
                        Classified
                      </div>
                    </div>

                    <div className="decision-stat-grid">
                      <div className="decision-stat match-stat">
                        <span>Automatic matches</span>
                        <strong>
                          {decisionResult.match_count.toLocaleString()}
                        </strong>
                        <small>
                          ≥ 90% probability
                        </small>
                      </div>

                      <div className="decision-stat possible-stat">
                        <span>Possible matches</span>
                        <strong>
                          {decisionResult.possible_match_count.toLocaleString()}
                        </strong>
                        <small>
                          50% – &lt; 90%
                        </small>
                      </div>

                      <div className="decision-stat distinct-stat">
                        <span>Non-matches</span>
                        <strong>
                          {decisionResult.non_match_count.toLocaleString()}
                        </strong>
                        <small>
                          &lt; 50% probability
                        </small>
                      </div>
                    </div>
                  </div>
                )}

                {/* -------- Human Review -------- */}

                {loadingHumanReview && (
                  <div className="message processing">
                    <strong>
                      Loading Human Review queue...
                    </strong>

                    <span>
                      Preparing possible matches that
                      require human validation.
                    </span>
                  </div>
                )}

                {humanReviewResult && (
                  <div className="human-review-section">
                    <div className="subsection-heading">
                      <div>
                        <p className="eyebrow">
                          HUMAN REVIEW
                        </p>

                        <h4>
                          Validate Possible Matches
                        </h4>
                      </div>

                      <div className="schema-count">
                        {reviewCount}{" "}
                        {reviewCount === 1
                          ? "Pair"
                          : "Pairs"}{" "}
                        Remaining
                      </div>
                    </div>

                    <p className="schema-description">
                      Pairs with intermediate match
                      probabilities are reviewed manually
                      before entity clustering and golden
                      record creation.
                    </p>

                    <div className="review-progress">
                      <div className="review-stat">
                        <span>
                          Possible matches
                        </span>

                        <strong>
                          {reviewCount}
                        </strong>
                      </div>

                      <div className="review-stat reviewed-match-stat">
                        <span>
                          Confirmed matches
                        </span>

                        <strong>
                          {reviewedMatchCount}
                        </strong>
                      </div>

                      <div className="review-stat reviewed-distinct-stat">
                        <span>
                          Rejected matches
                        </span>

                        <strong>
                          {reviewedNonMatchCount}
                        </strong>
                      </div>
                    </div>

                    {currentReviewItem && (
                      <div className="pair-card review-pair-card">
                        <div className="pair-heading">
                          <div>
                            <p className="eyebrow">
                              POSSIBLE MATCH
                            </p>

                            <h4>
                              Do these records represent
                              the same entity?
                            </h4>
                          </div>

                          <div className="review-probability">
                            <span>
                              Match probability
                            </span>

                            <strong>
                              {(
                                currentReviewItem.match_probability *
                                100
                              ).toFixed(1)}
                              %
                            </strong>
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
                                  currentReviewItem
                                    .record_a
                                    .record_id
                                }
                              </strong>
                            </div>

                            <div className="record-fields">
                              {Object.entries(
                                currentReviewItem
                                  .record_a
                                  .data
                              ).map(
                                ([field, value]) => (
                                  <div
                                    className="record-field"
                                    key={`review-a-${field}`}
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
                                  currentReviewItem
                                    .record_b
                                    .record_id
                                }
                              </strong>
                            </div>

                            <div className="record-fields">
                              {Object.entries(
                                currentReviewItem
                                  .record_b
                                  .data
                              ).map(
                                ([field, value]) => (
                                  <div
                                    className="record-field"
                                    key={`review-b-${field}`}
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
                              handleHumanReview(
                                "non_match"
                              )
                            }
                            disabled={
                              submittingHumanReview
                            }
                          >
                            <span>×</span>
                            Non-Match
                          </button>

                          <button
                            type="button"
                            className="label-button match-button"
                            onClick={() =>
                              handleHumanReview(
                                "match"
                              )
                            }
                            disabled={
                              submittingHumanReview
                            }
                          >
                            <span>✓</span>
                            Confirm Match
                          </button>
                        </div>
                      </div>
                    )}

                    {submittingHumanReview && (
                      <div className="message processing">
                        <strong>
                          Saving review decision...
                        </strong>

                        <span>
                          Recording the human validation
                          before moving to the next pair.
                        </span>
                      </div>
                    )}

                    {humanReviewMessage && (
                      <div className="message success">
                        <strong>
                          Review decision saved.
                        </strong>

                        <span>
                          {humanReviewMessage}
                        </span>
                      </div>
                    )}

                    {humanReviewComplete && (
                      <div className="ready-banner">
                        <div className="ready-icon">
                          ✓
                        </div>

                        <div>
                          <strong>
                            Human Review complete
                          </strong>

                          <span>
                            All possible matches have been
                            manually resolved. The pipeline
                            is ready for entity clustering
                            and golden record creation.
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* -------- Entity Clustering -------- */}

                {humanReviewComplete &&
                  !clusteringResult && (
                    <div className="entity-clustering-start">
                      <div className="entity-clustering-start-icon">
                        12
                      </div>

                      <div className="entity-clustering-start-content">
                        <strong>
                          Build Entity Clusters
                        </strong>

                        <span>
                          Group confirmed duplicate records
                          into unified entities using
                          automatic matches and
                          human-reviewed decisions.
                        </span>
                      </div>

                      <button
                        type="button"
                        className="upload-button"
                        onClick={
                          handleEntityClustering
                        }
                        disabled={processing}
                      >
                        {clusteringEntities
                          ? "Clustering..."
                          : "Build Entity Clusters"}
                      </button>
                    </div>
                  )}

                {clusteringEntities && (
                  <div className="message processing">
                    <strong>
                      Building entity clusters...
                    </strong>

                    <span>
                      Grouping confirmed duplicate records
                      using transitive entity relationships.
                    </span>
                  </div>
                )}

                {clusteringResult && (
                  <div className="entity-clustering-section">
                    <div className="subsection-heading">
                      <div>
                        <p className="eyebrow">
                          ENTITY CLUSTERING COMPLETE
                        </p>

                        <h4>
                          Resolved Entity Groups
                        </h4>
                      </div>

                      <div className="schema-count">
                        {clusteringResult.cluster_count}{" "}
                        {clusteringResult.cluster_count === 1
                          ? "Entity"
                          : "Entities"}
                      </div>
                    </div>

                    <p className="schema-description">
                      Confirmed duplicate records have been
                      grouped into unified entity clusters.
                      Records connected through matching
                      relationships are resolved into the
                      same entity.
                    </p>

                    <div className="cluster-stat-grid">
                      <div className="cluster-stat">
                        <span>
                          Entity clusters
                        </span>

                        <strong>
                          {clusteringResult.cluster_count.toLocaleString()}
                        </strong>

                        <small>
                          Resolved duplicate entities
                        </small>
                      </div>

                      <div className="cluster-stat">
                        <span>
                          Clustered records
                        </span>

                        <strong>
                          {clusteringResult.clustered_record_count.toLocaleString()}
                        </strong>

                        <small>
                          Records assigned to entities
                        </small>
                      </div>

                      <div className="cluster-stat">
                        <span>
                          Unclustered records
                        </span>

                        <strong>
                          {clusteringResult.unclustered_record_count.toLocaleString()}
                        </strong>

                        <small>
                          No confirmed duplicate relationship
                        </small>
                      </div>
                    </div>

                    <div className="cluster-list">
                      {clusteringResult.clusters.map(
                        (cluster) => (
                          <div
                            className="cluster-card"
                            key={cluster.cluster_id}
                          >
                            <div className="cluster-card-header">
                              <div className="cluster-identity">
                                <div className="cluster-number">
                                  {String(
                                    cluster.cluster_id
                                  ).padStart(2, "0")}
                                </div>

                                <div>
                                  <strong>
                                    Entity Cluster{" "}
                                    {cluster.cluster_id}
                                  </strong>

                                  <span>
                                    {
                                      cluster.record_ids
                                        .length
                                    }{" "}
                                    records
                                  </span>
                                </div>
                              </div>

                              <span className="cluster-badge">
                                Duplicate entity
                              </span>
                            </div>

                            <div className="cluster-records">
                              {cluster.record_ids.map(
                                (recordId) => (
                                  <span
                                    className="cluster-record"
                                    key={recordId}
                                  >
                                    Record #{recordId}
                                  </span>
                                )
                              )}
                            </div>
                          </div>
                        )
                      )}
                    </div>

                    {clusteringMessage && (
                      <div className="message success">
                        <strong>
                          Entity clustering completed.
                        </strong>

                        <span>
                          {
                            clusteringResult.cluster_count
                          }{" "}
                          duplicate entities were resolved
                          from{" "}
                          {
                            clusteringResult.clustered_record_count
                          }{" "}
                          records.
                        </span>
                      </div>
                    )}

                    <div className="clustering-complete-banner">
                      <div className="clustering-complete-icon">
                        ✓
                      </div>

                      <div>
                        <strong>
                          Entity Resolution Complete
                        </strong>

                        <span>
                          Duplicate records have been
                          grouped into entity-level
                          clusters. The pipeline is now
                          ready for golden record creation
                          and survivorship.
                        </span>
                      </div>
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

