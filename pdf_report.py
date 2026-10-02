"""Generate a compact PDF report from a local inference result and upload."""

from datetime import datetime
from html import escape
from io import BytesIO

from PIL import Image as PILImage, ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


INK = colors.HexColor('#17232D')
MUTED = colors.HexColor('#58717E')
TEAL = colors.HexColor('#138A72')
PALE = colors.HexColor('#EDF6F3')


def report_suggestions(result):
    if not result.get('classification_permitted'):
        return [
            'No DR grade was produced for this image.',
            'Try a clear colour fundus photograph with the retinal area fully visible.',
            'Ask a qualified eye-care professional to review concerns about your vision.',
        ]
    grade = int(result['predicted_grade'])
    if grade == 0:
        first = 'The model estimated No DR. Continue regular comprehensive eye examinations if you have diabetes.'
    elif grade in (1, 2):
        first = 'The model estimated diabetic retinopathy changes. Share the image and this report with an eye-care professional.'
    else:
        first = 'The model estimated a more advanced grade. Arrange prompt review by an eye-care professional.'
    return [
        first,
        'A qualified clinician can confirm the finding with an eye examination.',
        'Do not use this prototype result alone to make treatment decisions.',
    ]


def _paragraph(value, style):
    return Paragraph(escape(str(value)), style)


def _image_flowable(payload):
    with PILImage.open(BytesIO(payload)) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    image.thumbnail((1100, 900), PILImage.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=86, optimize=True)
    buffer.seek(0)
    width, height = ImageReader(buffer).getSize()
    ratio = min(228 / width, 195 / height)
    return Image(buffer, width=width * ratio, height=height * ratio)


def build_pdf_report(result, payload):
    output = BytesIO()
    document = SimpleDocTemplate(
        output, pagesize=A4, leftMargin=42, rightMargin=42,
        topMargin=38, bottomMargin=42, title='Retinal image analysis report',
        author='RetiScreen AI coursework prototype',
    )
    normal = ParagraphStyle('normal', fontName='Helvetica', fontSize=9.5,
                            leading=14, textColor=INK, spaceAfter=0)
    small = ParagraphStyle('small', parent=normal, fontSize=8.5, leading=12,
                           textColor=MUTED)
    kicker = ParagraphStyle('kicker', parent=small, fontName='Helvetica-Bold',
                            textColor=TEAL, tracking=1)
    title = ParagraphStyle('title', parent=normal, fontName='Helvetica-Bold',
                           fontSize=21, leading=25, spaceAfter=9)
    heading = ParagraphStyle('heading', parent=normal, fontName='Helvetica-Bold',
                             fontSize=12, leading=16, spaceBefore=16, spaceAfter=9)
    estimate = ParagraphStyle('estimate', parent=normal, fontName='Helvetica-Bold',
                              fontSize=19, leading=23, textColor=TEAL)

    filename = result.get('filename') or 'Uploaded image'
    date = datetime.now().astimezone().strftime('%d %B %Y')
    story = [
        Paragraph('RETISCREEN AI  /  RESEARCH PROTOTYPE', kicker),
        Spacer(1, 8),
        Paragraph('Retinal image analysis report', title),
        _paragraph(f'Image: {filename}    |    Generated: {date}', small),
        Spacer(1, 21),
    ]

    if result.get('classification_permitted'):
        result_text = [
            Paragraph('MODEL ESTIMATE', kicker),
            Spacer(1, 9),
            _paragraph(f"Grade {result['predicted_grade']} - {result['predicted_class']}", estimate),
            Spacer(1, 9),
            _paragraph(f"Model probability: {result['confidence'] * 100:.1f}%", normal),
            Spacer(1, 7),
            _paragraph('Review recommended' if result['status'] == 'review_required'
                       else 'Passed prototype routing checks', small),
        ]
    else:
        result_text = [
            Paragraph('NO GRADE PRODUCED', kicker),
            Spacer(1, 9),
            Paragraph('Image not graded', estimate),
            Spacer(1, 9),
            _paragraph('The upload did not pass the prototype image checks.', normal),
        ]

    overview = Table([[ _image_flowable(payload), result_text ]],
                     colWidths=[252, 258], hAlign='LEFT')
    overview.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PALE),
        ('BOX', (0, 0), (-1, -1), .5, colors.HexColor('#D2E2DC')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.extend([overview, Paragraph('Suggested next steps', heading)])
    for index, suggestion in enumerate(report_suggestions(result), 1):
        row = Table([[
            _paragraph(f'{index}.', kicker), _paragraph(suggestion, normal)
        ]], colWidths=[21, 489], hAlign='LEFT')
        row.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(row)

    story.extend([
        Spacer(1, 16),
        _paragraph('This coursework prototype is not a diagnosis or medical device. '
                   'The image check and model estimate need professional interpretation.', small),
        Spacer(1, 8),
        _paragraph('General eye-care guidance: National Eye Institute, Diabetic Retinopathy - '
                   'https://www.nei.nih.gov/eye-health-information/eye-conditions-and-diseases/diabetic-retinopathy', small),
    ])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#D7E3E0'))
        canvas.line(42, 31, A4[0] - 42, 31)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(42, 19, 'RetiScreen AI - Local research prototype')
        canvas.drawRightString(A4[0] - 42, 19, f'Page {doc.page}')
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()
