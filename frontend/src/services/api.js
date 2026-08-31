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

