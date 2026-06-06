const CARDIO_API_URL = PropertiesService.getScriptProperties().getProperty("CARDIO_API_URL") || "https://your-api-url/predict";

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu("Cardio Catch")
    .addItem("Score active sheet", "scoreActiveSheet")
    .addToUi();
}

function scoreActiveSheet() {
  const sheet = SpreadsheetApp.getActiveSheet();
  const values = sheet.getDataRange().getValues();
  const headers = values[0];
  const records = values.slice(1).filter(row => row.join("") !== "").map(row => {
    const record = {};
    headers.forEach((header, index) => record[header] = row[index]);
    return record;
  });

  const response = UrlFetchApp.fetch(CARDIO_API_URL, {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify({records: records}),
    muteHttpExceptions: true
  });

  if (response.getResponseCode() >= 300) {
    throw new Error(response.getContentText());
  }

  const payload = JSON.parse(response.getContentText());
  const predictionColumn = headers.length + 1;
  const scoreColumn = headers.length + 2;
  sheet.getRange(1, predictionColumn).setValue("prediction");
  sheet.getRange(1, scoreColumn).setValue("risk_score");

  payload.prediction.forEach((prediction, index) => {
    sheet.getRange(index + 2, predictionColumn).setValue(prediction);
    if (payload.score) {
      sheet.getRange(index + 2, scoreColumn).setValue(payload.score[index]);
    }
  });
}
