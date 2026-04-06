from datetime import datetime, timezone
from pathlib import Path
from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from app.core.config import get_settings
from app.models.report import Report
from app.models.report_feedback import ReportFeedback
from app.models.training_sample import TrainingSample
from app.models.user import User
from app.schemas.report import (
    ReportFeedbackRequest,
    ReportFeedbackResponse,
    ReportResponse,
    SimplifyTextRequest,
    UploadResponse,
)
from app.services.grounding import build_grounded_points
from app.services.glossary import build_glossary_entries, update_term_memory
from app.services.ocr import OCRUnavailableError, extract_text_from_file
from app.services.safety import detect_clinical_safety_alerts
from app.services.simplify import extract_important_terms, simplify_dual_output
from app.utils.deps import get_current_user

router = APIRouter(prefix="/reports", tags=["reports"])
settings = get_settings()


def report_to_response(report: Report) -> ReportResponse:
    return ReportResponse(
        id=str(report.id),
        user_id=str(report.user_id),
        file_name=report.file_name,
        extracted_text=report.extracted_text,
        simplified_text=report.simplified_text,
        caregiver_text=report.caregiver_text or report.simplified_text,
        important_terms=report.important_terms,
        glossary_entries=report.glossary_entries or [],
        safety_alerts=report.safety_alerts or [],
        grounded_points=report.grounded_points or [],
        created_at=report.created_at
    )


@router.post("/simplify-text", response_model=UploadResponse)
async def simplify_raw_text(
    payload: SimplifyTextRequest,
    save_result: bool = False,
    current_user: User = Depends(get_current_user)
):
    extracted_text = payload.text.strip()
    patient_text, caregiver_text = simplify_dual_output(extracted_text)
    simplified_text = patient_text
    terms = extract_important_terms(extracted_text)
    glossary_entries = build_glossary_entries(extracted_text, terms)
    await update_term_memory(glossary_entries)
    safety_alerts = detect_clinical_safety_alerts(extracted_text)
    grounded_points = build_grounded_points(extracted_text, simplified_text)

    if save_result:
        report = Report(
            user_id=current_user.id,
            file_name="manual_text_entry.txt",
            extracted_text=extracted_text,
            simplified_text=simplified_text,
            caregiver_text=caregiver_text,
            important_terms=terms,
            glossary_entries=glossary_entries,
            safety_alerts=safety_alerts,
            grounded_points=grounded_points,
        )
        await report.insert()

        if extracted_text.strip() and simplified_text.strip():
            training_sample = TrainingSample(
                user_id=current_user.id,
                report_id=report.id,
                source_text=extracted_text,
                target_text=simplified_text,
                caregiver_target_text=caregiver_text,
                source_type="manual_text",
                sample_quality="generated",
                is_shared=True,
            )
            await training_sample.insert()

        created_at = report.created_at
        report_response = report_to_response(report)
    else:
        report_response = None
        created_at = datetime.now(timezone.utc)

    return UploadResponse(
        saved=save_result,
        report=report_response,
        file_name="manual_text_entry.txt",
        extracted_text=extracted_text,
        simplified_text=simplified_text,
        caregiver_text=caregiver_text,
        important_terms=terms,
        glossary_entries=glossary_entries,
        safety_alerts=safety_alerts,
        grounded_points=grounded_points,
        created_at=created_at
    )


@router.get("", response_model=list[ReportResponse])
async def list_reports(current_user: User = Depends(get_current_user)):
    reports = await Report.find(
        Report.user_id == current_user.id
    ).sort(-Report.created_at).to_list()
    return [report_to_response(r) for r in reports]


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: str,
    current_user: User = Depends(get_current_user)
):
    try:
        obj_id = PydanticObjectId(report_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid report ID")

    report = await Report.find_one(
        Report.id == obj_id,
        Report.user_id == current_user.id
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report_to_response(report)


@router.post("/{report_id}/feedback", response_model=ReportFeedbackResponse)
async def submit_report_feedback(
    report_id: str,
    payload: ReportFeedbackRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        obj_id = PydanticObjectId(report_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid report ID")

    report = await Report.find_one(
        Report.id == obj_id,
        Report.user_id == current_user.id,
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if not (1 <= payload.clarity_rating <= 5) or not (1 <= payload.accuracy_rating <= 5):
        raise HTTPException(status_code=400, detail="Ratings must be between 1 and 5")

    feedback = ReportFeedback(
        user_id=current_user.id,
        report_id=report.id,
        clarity_rating=payload.clarity_rating,
        accuracy_rating=payload.accuracy_rating,
        corrected_text=payload.corrected_text.strip(),
        comment=payload.comment.strip(),
    )
    await feedback.insert()

    corrected_text = payload.corrected_text.strip()
    existing_sample = await TrainingSample.find_one(TrainingSample.report_id == report.id)
    if existing_sample:
        existing_sample.clarity_rating = payload.clarity_rating
        existing_sample.accuracy_rating = payload.accuracy_rating
        existing_sample.is_shared = True
        if corrected_text:
            existing_sample.corrected_target_text = corrected_text
            existing_sample.source_type = "user_feedback"
            existing_sample.sample_quality = "feedback_corrected"
        else:
            existing_sample.sample_quality = "rated_generated"

        if not existing_sample.target_text and report.simplified_text.strip():
            existing_sample.target_text = report.simplified_text
        if not existing_sample.caregiver_target_text and report.caregiver_text.strip():
            existing_sample.caregiver_target_text = report.caregiver_text

        await existing_sample.save()
    elif report.extracted_text.strip() and report.simplified_text.strip():
        sample = TrainingSample(
            user_id=current_user.id,
            report_id=report.id,
            source_text=report.extracted_text,
            target_text=report.simplified_text,
            caregiver_target_text=report.caregiver_text,
            corrected_target_text=corrected_text,
            clarity_rating=payload.clarity_rating,
            accuracy_rating=payload.accuracy_rating,
            source_type="user_feedback" if corrected_text else "feedback_rating",
            sample_quality="feedback_corrected" if corrected_text else "rated_generated",
            is_shared=True,
        )
        await sample.insert()

    return ReportFeedbackResponse(message="Feedback saved")


@router.post("/upload", response_model=UploadResponse)
async def upload_report(
    file: UploadFile = File(...),
    save_result: bool = Form(True),
    current_user: User = Depends(get_current_user)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid file")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".pdf"}:
        raise HTTPException(status_code=400, detail="Only PNG, JPG, JPEG, and PDF are supported")

    file_bytes = await file.read()
    try:
        extracted_text = extract_text_from_file(file.filename, file_bytes)
    except OCRUnavailableError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{exc} Use /reports/simplify-text for manual text input."
        ) from exc

    patient_text, caregiver_text = simplify_dual_output(extracted_text)
    simplified_text = patient_text
    terms = extract_important_terms(extracted_text)
    glossary_entries = build_glossary_entries(extracted_text, terms)
    await update_term_memory(glossary_entries)
    safety_alerts = detect_clinical_safety_alerts(extracted_text)
    grounded_points = build_grounded_points(extracted_text, simplified_text)

    if save_result:
        report = Report(
            user_id=current_user.id,
            file_name=file.filename,
            extracted_text=extracted_text or "",
            simplified_text=simplified_text,
            caregiver_text=caregiver_text,
            important_terms=terms,
            glossary_entries=glossary_entries,
            safety_alerts=safety_alerts,
            grounded_points=grounded_points,
        )
        await report.insert()

        if extracted_text.strip() and simplified_text.strip():
            training_sample = TrainingSample(
                user_id=current_user.id,
                report_id=report.id,
                source_text=extracted_text,
                target_text=simplified_text,
                caregiver_target_text=caregiver_text,
                source_type="file_upload",
                sample_quality="generated",
                is_shared=True,
            )
            await training_sample.insert()

        created_at = report.created_at
        report_response = report_to_response(report)
    else:
        report_response = None
        created_at = datetime.now(timezone.utc)

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    (upload_dir / safe_name).write_bytes(file_bytes)

    return UploadResponse(
        saved=save_result,
        report=report_response,
        file_name=file.filename,
        extracted_text=extracted_text,
        simplified_text=simplified_text,
        caregiver_text=caregiver_text,
        important_terms=terms,
        glossary_entries=glossary_entries,
        safety_alerts=safety_alerts,
        grounded_points=grounded_points,
        created_at=created_at
    )
