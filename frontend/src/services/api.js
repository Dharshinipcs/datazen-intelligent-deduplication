const API_BASE_URL = "";

async function parseResponse(response, fallbackMessage) {
  let data;

  try {
    data = await response.json();
  } catch {
    throw new Error("The server returned an invalid response.");
  }

  if (!response.ok) {
    const message =
      typeof data.detail === "string"
        ? data.detail
        : fallbackMessage;

    throw new Error(message);
  }

  return data;
}

export async function uploadDataset(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(
    `${API_BASE_URL}/api/datasets/upload`,
    {
      method: "POST",
      body: formData,
    }
  );

  return parseResponse(response, "Dataset upload failed.");
}

export async function profileDataset(datasetId) {
  const response = await fetch(
    `${API_BASE_URL}/api/datasets/${datasetId}/profile`,
    {
      method: "POST",
    }
  );

  return parseResponse(response, "Dataset profiling failed.");
}

export async function detectSchema(datasetId) {
  const response = await fetch(
    `${API_BASE_URL}/api/datasets/${datasetId}/schema`,
    {
      method: "POST",
    }
  );

  return parseResponse(response, "Schema detection failed.");
}

export async function standardizeDataset(datasetId) {
  const response = await fetch(
    `${API_BASE_URL}/api/datasets/${datasetId}/standardize`,
    {
      method: "POST",
    }
  );

  return parseResponse(
    response,
    "Dataset standardization failed."
  );
}

/* ---------------- DEDUPE ---------------- */

export async function prepareDedupe(datasetId) {
  const response = await fetch(
    `${API_BASE_URL}/api/datasets/${datasetId}/dedupe/prepare`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        sample_size: 1500,
        blocked_proportion: 0.9,
      }),
    }
  );

  return parseResponse(
    response,
    "Failed to prepare dataset for deduplication."
  );
}

export async function getUncertainDedupePair(datasetId) {
  const response = await fetch(
    `${API_BASE_URL}/api/datasets/${datasetId}/dedupe/uncertain-pairs?limit=1`,
    {
      method: "GET",
    }
  );

  return parseResponse(
    response,
    "Failed to fetch the next uncertain pair."
  );
}

export async function labelDedupePair(datasetId, pair, label) {
  const payload =
    label === "match"
      ? {
          matches: [pair],
          distinct: [],
        }
      : {
          matches: [],
          distinct: [pair],
        };

  const response = await fetch(
    `${API_BASE_URL}/api/datasets/${datasetId}/dedupe/label`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    }
  );

  return parseResponse(
    response,
    "Failed to save the dedupe label."
  );
}

export async function trainDedupe(datasetId) {
  const response = await fetch(
    `${API_BASE_URL}/api/datasets/${datasetId}/dedupe/train`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        recall: 1.0,
        index_predicates: true,
      }),
    }
  );

  return parseResponse(
    response,
    "Failed to train the deduplication model."
  );
}

export async function generateBlockingStrategy(datasetId) {
  const response = await fetch(
    `${API_BASE_URL}/api/datasets/${datasetId}/dedupe/blocking-strategy/generate`,
    {
      method: "POST",
    }
  );

  return parseResponse(
    response,
    "Failed to generate the blocking strategy."
  );
}