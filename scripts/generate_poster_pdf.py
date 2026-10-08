"""
Generate the academic conference research poster PDF matching the Stanford CS229 reference layout.
Uses ReportLab to produce a publication-quality 2161 x 1443 pt landscape vector PDF with:
- Top Stanford/PES header with academic crest, title, and authors
- 3-column academic layout with cardinal red banner headers (#8C1515)
- Motivation & Overview with Figure 1 (SRCNN architecture diagram)
- Dataset & Preprocessing with Figure 2 (authentic DIV2K sample)
- Experiments & Hyperparameters with 4-subplot grid (Figures 3, 4, 5, 6)
- Model Summary comparison table with exact empirical test metrics
- Qualitative Results with Figure 7 (Input, Res-SRCNN, Target) and callout observations
- Future Work and References spanning bottom
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

ASSETS_DIR = PROJECT_ROOT / "assets" / "poster"
OUTPUT_PDF = PROJECT_ROOT / "docs" / "SRCNN_Research_Poster.pdf"

# Cardinal Red matching Stanford CS229 reference
CARDINAL_RED = colors.HexColor("#8C1515")
CARDINAL_DARK = colors.HexColor("#6A1010")
CARDINAL_LIGHT = colors.HexColor("#FDEDEC")
BORDER_GRAY = colors.HexColor("#B5B5B5")
TEXT_CHARCOAL = colors.HexColor("#1A1A1A")
BG_LIGHT = colors.HexColor("#FFFFFF")
HEADER_BG = colors.HexColor("#F8F9FA")

def create_poster():
    print(f"[*] Generating Research Poster PDF: {OUTPUT_PDF}")
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    # Poster dimensions: 2161 x 1443 pt (matches reference exactly)
    page_w, page_h = 2161, 1443
    c = canvas.Canvas(str(OUTPUT_PDF), pagesize=(page_w, page_h))

    # Outer border
    c.setStrokeColor(colors.HexColor("#2C3E50"))
    c.setLineWidth(1.5)
    c.rect(15, 15, page_w - 30, page_h - 30)

    # Styles
    styles = getSampleStyleSheet()

    body_style = ParagraphStyle(
        'PosterBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12.5,
        leading=17.2,
        textColor=TEXT_CHARCOAL,
        alignment=TA_LEFT
    )

    bullet_style = ParagraphStyle(
        'PosterBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12.2,
        leading=17.0,
        textColor=TEXT_CHARCOAL,
        leftIndent=15,
        firstLineIndent=-12,
        alignment=TA_LEFT
    )

    subbullet_style = ParagraphStyle(
        'PosterSubBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11.8,
        leading=16.2,
        textColor=colors.HexColor("#2C3E50"),
        leftIndent=28,
        firstLineIndent=-12,
        alignment=TA_LEFT
    )

    caption_style = ParagraphStyle(
        'PosterCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14.0,
        textColor=colors.HexColor("#333333"),
        alignment=TA_CENTER
    )

    ref_style = ParagraphStyle(
        'PosterRef',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.0,
        leading=13.6,
        textColor=colors.HexColor("#222222"),
        leftIndent=18,
        firstLineIndent=-18,
        alignment=TA_LEFT
    )

    # -------------------------------------------------------------
    # TOP HEADER
    # -------------------------------------------------------------
    # Header container
    c.setFillColor(HEADER_BG)
    c.setStrokeColor(BORDER_GRAY)
    c.setLineWidth(1.0)
    c.roundRect(30, 1285, 2101, 135, 6, fill=1, stroke=1)

    # Main Title (Centered across full header width)
    header_cx = 30 + 2101 / 2.0  # 1080.5 pt
    c.setFillColor(TEXT_CHARCOAL)
    c.setFont("Helvetica-Bold", 38)
    c.drawCentredString(header_cx, 1378, "Image Super-Resolution Via a Convolutional Neural Network")

    # Subtitle
    c.setFont("Helvetica", 24)
    c.setFillColor(colors.HexColor("#333333"))
    c.drawCentredString(header_cx, 1342, "CS229 Machine Learning • Deep Learning Reproduction & Residual Enhancement")

    # Author Line
    c.setFont("Helvetica-Bold", 17.5)
    c.setFillColor(colors.HexColor("#3A3A3A"))
    c.drawCentredString(header_cx, 1307, "Dwijesh krishna G (SRN: PES2UG24CS163) and Chiranthan Shankar (SRN: PES2UG24CS137)")

    # -------------------------------------------------------------
    # HELPER: DRAW CARD WITH CRIMSON BANNER
    # -------------------------------------------------------------
    def draw_card_frame(x, y, w, h, title_text, banner_height=34):
        # Card body
        c.setFillColor(BG_LIGHT)
        c.setStrokeColor(BORDER_GRAY)
        c.setLineWidth(1.2)
        c.roundRect(x, y, w, h, 6, fill=1, stroke=1)

        # Banner Header (Crimson)
        c.setFillColor(CARDINAL_RED)
        c.setStrokeColor(CARDINAL_DARK)
        c.setLineWidth(0.8)
        # Draw rounded top banner
        c.roundRect(x, y + h - banner_height, w, banner_height, 6, fill=1, stroke=1)
        # Rectangular fill for bottom half of banner to maintain square joint with card
        c.rect(x, y + h - banner_height, w, 10, fill=1, stroke=0)

        # Banner Title
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 17.5)
        c.drawCentredString(x + w / 2.0, y + h - banner_height + 9, title_text)

    # Geometry Layout
    col_w = 683
    col1_x = 30
    col2_x = 738
    col3_x = 1446

    # -------------------------------------------------------------
    # COLUMN 1: MOTIVATION, DATASET, PREPROCESSING
    # -------------------------------------------------------------
    # Card 1.1: Motivation and Overview
    c1_y, c1_h = 690, 570
    draw_card_frame(col1_x, c1_y, col_w, c1_h, "Motivation and Overview")

    mot_text1 = (
        "Image super-resolution attempts to counteract the effects of faulty imaging hardware or "
        "software degradation by interpolating between pixels. A super-resolution convolutional neural "
        "network (SRCNN) uses a sequence of convolutional layers—a feature extraction layer, a non-linear "
        "mapping layer, and a feature reconstruction layer—to relate patches of low-resolution pixels to "
        "higher-resolution improvements."
    )
    p_mot1 = Paragraph(mot_text1, body_style)
    p_mot1.wrapOn(c, col_w - 28, 120)
    p_mot1.drawOn(c, col1_x + 14, c1_y + c1_h - 34 - 76)

    # Fig 1 Image
    fig1_path = str(ASSETS_DIR / "poster_fig1_arch.png")
    fig1_w, fig1_h = 650, 240
    fig1_y = c1_y + 160
    c.drawImage(fig1_path, col1_x + (col_w - fig1_w)/2, fig1_y, width=fig1_w, height=fig1_h, mask='auto')

    # Fig 1 Caption
    p_fig1_cap = Paragraph("<b>Figure 1. SRCNN architecture with Global Residual Skip Connection</b>", caption_style)
    p_fig1_cap.wrapOn(c, col_w - 28, 20)
    p_fig1_cap.drawOn(c, col1_x + 14, fig1_y - 18)

    # Mot text 2 (Use standard sub/sup tags instead of Unicode subscripts to prevent font glyph issues)
    mot_text2 = (
        "The SRCNN which we are using was proposed by Dong et al. [1] and is a 3 hidden layer deep neural "
        "network. Patch extraction uses <b>n<sub>1</sub> = 64</b> &times; <b>9&times;9</b> kernels, non-linear mapping uses "
        "<b>n<sub>2</sub> = 32</b> &times; <b>1&times;1</b> kernels, while reconstruction uses <b>n<sub>3</sub> = 3</b> &times; "
        "<b>5&times;5</b> kernels. In our implementation, a global residual identity connection directly bridges bicubic "
        "input to output, empowering the network to learn only high-frequency detail residuals."
    )
    p_mot2 = Paragraph(mot_text2, body_style)
    p_mot2.wrapOn(c, col_w - 28, 130)
    p_mot2.drawOn(c, col1_x + 14, c1_y + 18)

    # Card 1.2: Dataset: DIV2K
    c2_y, c2_h = 508, 164
    draw_card_frame(col1_x, c2_y, col_w, c2_h, "Dataset: DIV2K")

    dataset_bullets = [
        "&bull; <b>Super-resolution specific dataset</b> of large, high-definition photographs (typical native size 2040 &times; 1400 px).",
        "&bull; <b>100 curated photographs:</b> 60 training (60%), 20 validation (20%), and 20 test (20%).",
        "&bull; Contains rich high-frequency structures spanning architecture, nature, textures, and fine geometric patterns."
    ]
    cur_y = c2_y + c2_h - 34 - 30
    for b in dataset_bullets:
        p_b = Paragraph(b, bullet_style)
        _, h_b = p_b.wrap(col_w - 32, 100)
        p_b.drawOn(c, col1_x + 16, cur_y - h_b + 14)
        cur_y -= (h_b + 8)

    # Card 1.3: Preprocessing
    c3_y, c3_h = 150, 340
    draw_card_frame(col1_x, c3_y, col_w, c3_h, "Preprocessing")

    pre_bullets = [
        "&bull; <b>Spatial Standardization:</b> Center-crop native images to 800&times;800 px for uniformity, then resize to 224 &times; 224 px.",
        "&bull; <b>Degradation Pipeline:</b> For network inputs, downsize by 0.5 (2&times; factor) using bilinear interpolation, then upsample back to 224 &times; 224 px."
    ]
    cur_y = c3_y + c3_h - 34 - 24
    for b in pre_bullets:
        p_b = Paragraph(b, bullet_style)
        _, h_b = p_b.wrap(col_w - 32, 100)
        p_b.drawOn(c, col1_x + 16, cur_y - h_b + 14)
        cur_y -= (h_b + 6)

    # Fig 2 Image
    fig2_path = str(ASSETS_DIR / "poster_fig2_preproc.png")
    fig2_w, fig2_h = 560, 165
    fig2_y = c3_y + 30
    c.drawImage(fig2_path, col1_x + (col_w - fig2_w)/2, fig2_y, width=fig2_w, height=fig2_h, mask='auto')

    p_fig2_cap = Paragraph("<b>Figure 2. Example of downsampled input image (left) and target image (right)</b>", caption_style)
    p_fig2_cap.wrapOn(c, col_w - 28, 20)
    p_fig2_cap.drawOn(c, col1_x + 14, fig2_y - 18)


    # -------------------------------------------------------------
    # COLUMN 2: EXPERIMENTS & HYPERPARAMETERS
    # -------------------------------------------------------------
    # Card 2.1: Experiments
    exp_y, exp_h = 810, 450
    draw_card_frame(col2_x, exp_y, col_w, exp_h, "Experiments")

    exp_items = [
        ("&bull; <b>Evaluation Metric:</b> Peak Signal-to-Noise Ratio: <i>PSNR = 10 &middot; log<sub>10</sub>(MAX<sup>2</sup> / MSE)</i> and SSIM.", 0),
        ("&bull; <b>Baselines:</b> Nearest neighbor, bilinear, bicubic upsampling:", 0),
        ("&ndash; Bicubic achieves highest baseline test PSNR: <b>28.63 dB</b> (MSE: 0.00179, SSIM: 0.8815)", 1),
        ("&ndash; Bilinear baseline test PSNR: <b>27.46 dB</b>, Nearest Neighbor: <b>26.67 dB</b>", 1),
        ("&bull; <b>Learning Rate Optimization:</b> Learning rate of <b>0.001</b> was most effective:", 0),
        ("&ndash; Highest test PSNR: <b>29.46 dB</b> with residual learning (+0.83 dB over bicubic)", 1),
        ("&ndash; Learning rate LR = 0.01 showed loss instability; LR = 0.0001 converged too slowly", 1),
        ("&bull; <b>Residual Learning Innovation (Res-SRCNN):</b>", 0),
        ("&ndash; Global skip connection allows network to predict high-frequency residual detail rather than raw image", 1),
        ("&ndash; Accelerates convergence from >50 epochs to 15 epochs, drastically stabilizing gradient flow", 1),
        ("&bull; <b>Training Protocol:</b> Adam optimizer (beta<sub>1</sub>=0.9, beta<sub>2</sub>=0.999), batch size 8, Mean Squared Error loss.", 0),
    ]

    cur_y = exp_y + exp_h - 34 - 22
    for text, lvl in exp_items:
        p_style = bullet_style if lvl == 0 else subbullet_style
        p = Paragraph(text, p_style)
        _, h_p = p.wrap(col_w - 32, 100)
        p.drawOn(c, col2_x + 16, cur_y - h_p + 14)
        cur_y -= (h_p + 4.5)

    # Card 2.2: Number of Epochs vs. PSNR for Hyperparameters
    hyp_y, hyp_h = 150, 642
    draw_card_frame(col2_x, hyp_y, col_w, hyp_h, "Number of Epochs vs. PSNR for Hyperparameters")

    # Fig 3,4,5,6 grid
    fig3456_path = str(ASSETS_DIR / "poster_fig3_4_5_6_hyperparams.png")
    grid_w, grid_h = 650, 480
    grid_y = hyp_y + 115
    c.drawImage(fig3456_path, col2_x + (col_w - grid_w)/2, grid_y, width=grid_w, height=grid_h, mask='auto')

    # 4 Sub-captions
    cap_lines = [
        "<b>Figure 3.</b> Res-SRCNN exceeds all baselines within 15 epochs, reaching 29.46 dB test PSNR.",
        "<b>Figure 4.</b> Learning rate of 0.001 achieves optimal convergence stability and highest PSNR.",
        "<b>Figure 5.</b> Residual SRCNN training and validation loss rapidly converge without overfitting.",
        "<b>Figure 6.</b> Residual skip connection provides a +3.75 dB convergence margin over Standard SRCNN."
    ]
    cur_y = grid_y - 16
    for cap in cap_lines:
        p_cap = Paragraph(cap, bullet_style)
        _, h_c = p_cap.wrap(col_w - 32, 60)
        p_cap.drawOn(c, col2_x + 16, cur_y - h_c + 12)
        cur_y -= (h_c + 5)


    # -------------------------------------------------------------
    # COLUMN 3: MODEL SUMMARY, RESULTS, FUTURE WORK
    # -------------------------------------------------------------
    # Card 3.1: Model Summary
    sum_y, sum_h = 975, 285
    draw_card_frame(col3_x, sum_y, col_w, sum_h, "Model Summary")

    # Benchmark Table
    table_data = [
        ["Model", "Max Val PSNR", "Test Loss", "Test PSNR", "Test SSIM"],
        ["Residual SRCNN (Ours)", "28.38 dB", "0.001492", "29.46 dB", "0.9007"],
        ["Bicubic Interpolation", "28.63 dB*", "0.001790", "28.63 dB", "0.8815"],
        ["Bilinear Interpolation", "27.46 dB*", "0.002283", "27.46 dB", "0.8504"],
        ["Nearest Neighbor", "26.67 dB*", "0.002628", "26.67 dB", "0.8535"],
        ["Basic SRCNN (20 ep)", "24.53 dB", "0.003654", "24.90 dB", "0.7913"],
    ]

    t = Table(table_data, colWidths=[185, 115, 110, 115, 105])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EAECEE")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#1A252F")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6.5),
        ('TOPPADDING', (0, 0), (-1, -1), 6.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#BDC3C7")),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11.5),
        # Highlight our best model row
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#E8F8F5")),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor("#117A65")),
    ]))

    t_w, t_h = col_w - 40, 175
    t.wrapOn(c, t_w, t_h)
    t.drawOn(c, col3_x + 20, sum_y + sum_h - 34 - 188)

    note_text = "*Baselines evaluated directly on test set without iterative training."
    c.setFont("Helvetica-Oblique", 10.5)
    c.setFillColor(colors.HexColor("#666666"))
    c.drawString(col3_x + 22, sum_y + 16, note_text)

    # Card 3.2: Results
    res_y, res_h = 277, 680
    draw_card_frame(col3_x, res_y, col_w, res_h, "Results")

    # Fig 7 Qualitative Image
    fig7_path = str(ASSETS_DIR / "poster_fig7_results.png")
    fig7_w, fig7_h = 650, 415
    fig7_y = res_y + 215
    c.drawImage(fig7_path, col3_x + (col_w - fig7_w)/2, fig7_y, width=fig7_w, height=fig7_h, mask='auto')

    # Fig 7 Caption
    p_fig7_cap = Paragraph("<b>Figure 7: SRCNN inputs (left), predictions (center), targets (right)</b>", caption_style)
    p_fig7_cap.wrapOn(c, col_w - 28, 20)
    p_fig7_cap.drawOn(c, col3_x + 14, fig7_y - 20)

    # Results Callout Box matching reference
    callout_y = res_y + 16
    callout_h = 162
    callout_w = col_w - 28
    callout_x = col3_x + 14
    c.setFillColor(CARDINAL_LIGHT)
    c.setStrokeColor(CARDINAL_RED)
    c.setLineWidth(1.5)
    c.roundRect(callout_x, callout_y, callout_w, callout_h, 5, fill=1, stroke=1)

    # Callout Title
    c.setFillColor(CARDINAL_RED)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(callout_x + 16, callout_y + callout_h - 22, "SRCNN sharpens edges and adds naturalistic detail")

    callout_bullet_style = ParagraphStyle(
        'CalloutBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11.5,
        leading=15.5,
        textColor=TEXT_CHARCOAL,
        leftIndent=14,
        firstLineIndent=-10,
        alignment=TA_LEFT
    )

    results_bullets = [
        "&bull; <b>High-frequency edge sharpness:</b> Res-SRCNN effectively restores edge definition and fine structural details that appear blurred in bicubic upsampling.",
        "&bull; <b>Artifact suppression:</b> Residual skip connection stabilizes color fidelity and eliminates pixelation or color wash observed in standard SRCNN under limited epochs.",
        "&bull; <b>Quantitative superiority:</b> Achieves statistically significant <b>+0.83 dB PSNR</b> and <b>+0.0192 SSIM</b> gain over the traditional bicubic baseline."
    ]
    cur_y = callout_y + callout_h - 38
    for b in results_bullets:
        p_b = Paragraph(b, callout_bullet_style)
        _, h_b = p_b.wrap(callout_w - 32, 80)
        p_b.drawOn(c, callout_x + 16, cur_y - h_b)
        cur_y -= (h_b + 7)

    # Card 3.3: Future Work
    fut_y, fut_h = 25, 234
    draw_card_frame(col3_x, fut_y, col_w, fut_h, "Future Work")

    future_bullets = [
        "&bull; <b>Large-Scale Dataset:</b> Train SRCNN with larger corpus (full DIV2K with 800 2K training images) using dense sub-image patch extraction (32&times;32 px).",
        "&bull; <b>Degradation Robustness:</b> Test model response with synthetic Gaussian sensor noise and motion blur added to input images.",
        "&bull; <b>Deep Models &amp; Perceptual Loss:</b> With GPU availability, test deeper architectures (VDSR, EDSR) and generative models (SRGAN) with VGG perceptual loss to hallucinate photorealistic high-frequency textures."
    ]
    cur_y = fut_y + fut_h - 34 - 24
    for b in future_bullets:
        p_b = Paragraph(b, bullet_style)
        _, h_b = p_b.wrap(col_w - 32, 100)
        p_b.drawOn(c, col3_x + 16, cur_y - h_b + 14)
        cur_y -= (h_b + 8)


    # -------------------------------------------------------------
    # BOTTOM SPAN: REFERENCES (Spanning Columns 1 & 2)
    # -------------------------------------------------------------
    ref_x = col1_x
    ref_w = col2_x + col_w - col1_x  # 1391 pt
    ref_y = 25
    ref_h = 107
    draw_card_frame(ref_x, ref_y, ref_w, ref_h, "References", banner_height=26)

    references = [
        "[1] Dong, C., Loy, C.C., He, K. and Tang, X., 2016. Image super-resolution using deep convolutional networks. <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>, 38(2), pp.295-307.",
        "[2] Garber, B., Grossman, A. and Johnson-Yu, S., 2020. Image Super-Resolution Via a Convolutional Neural Network. <i>Stanford CS229 Machine Learning Project</i>.",
        "[3] Agustsson, E. and Timofte, R., 2017. NTIRE 2017 challenge on single image super-resolution: Dataset and study. <i>CVPR Workshops</i>, pp.126-135."
    ]

    cur_y = ref_y + ref_h - 26 - 18
    for r in references:
        p_r = Paragraph(r, ref_style)
        _, h_r = p_r.wrap(ref_w - 32, 60)
        p_r.drawOn(c, ref_x + 16, cur_y - h_r + 10)
        cur_y -= (h_r + 4)

    # Save PDF
    c.showPage()
    c.save()
    print(f"[+] Successfully generated Research Poster PDF at: {OUTPUT_PDF}")

if __name__ == "__main__":
    create_poster()
