from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from pathlib import Path
from xml.sax.saxutils import escape

def create_report(s,comparison,audit,notes,importance,sensitivity):
    styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9,textColor=colors.HexColor('#243c4b')))
    story=[]; m=s['metrics']; t=s['targeting']
    def p(x): story.append(Paragraph(escape(str(x)),styles['BodyCustom']))
    def title(n,x):
        if n>1: story.append(PageBreak())
        story.append(Paragraph(f'{n:02d} / {x}',styles['Title'])); story.append(Spacer(1,12))
    def chart(name,w=470,h=215): story.append(Image('figures/'+name+'.png',width=w,height=h,kind='proportional')); story.append(Spacer(1,8))
    def table(data,widths=None):
        data=[[Paragraph(escape(str(v)),styles['BodyCustom']) for v in row] for row in data]
        tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT'); tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dceef0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#197f86')),('BOTTOMPADDING',(0,0),(-1,-1),4)])); story.append(tab)
    title(1,'Customer retention analytics')
    p('DECISION BRIEF | Subscription services | Reproducible analysis | 21 October 2026')
    p(f"The selected {s['winner']} model achieved held-out ROC-AUC {m['roc_auc']:.3f} and average precision {m['average_precision']:.3f}. The AP random-ranking benchmark is {m['test_prevalence']:.3f}; high prevalence makes apparently high AP and accuracy easier to obtain.")
    table([['Decision measure','Held-out result'],['Contact capacity',f"{int(t['contacted'])} / 240 customers"],['Churners captured',f"{int(t['true_churners_reached'])}; random expectation {t['random_expected_churners']:.1f}"],['Precision / recall at 20%',f"{t['precision_at_20']:.1%} / {t['recall_at_20']:.1%}"],['Lift over random',f"{t['lift']:.3f}x"],['Hypothetical incremental net value',f"${t['incremental_net_vs_random']:.0f}"]],[230,260])
    chart('gains',470,240)
    p('Use the ranking as a pilot outreach queue. Prediction identifies association, while incremental retention value requires a controlled experiment. Treat the unusually high churn prevalence as a sampling and target-definition validation priority.')
    title(2,'Data quality and preparation')
    p('Source: customer_churn_advanced_dataset.csv. 1,200 customers and 20 columns; 1,036 churn labels (86.33%) and 164 non-churn labels (13.67%). customer_id uniquely identifies every record. No exact duplicate rows were found.')
    table([['Missing predictor','Count','Percent']]+[[c,str(int(audit.loc[c,'missing'])),f"{audit.loc[c,'missing_pct']:.1f}%"] for c in ['monthly_charges','total_charges','satisfaction_score','monthly_usage_gb']],[260,100,130])
    p('The validated cleaned export preserves all records, identity, target labels and missing values. No full-dataset imputation is performed. Numeric medians and categorical modes are learned in each training fold. Scaling and unknown-safe one-hot encoding are contained in the pipeline.')
    p('Monthly charges range from 20 to 235.01; usage ranges from 2 to 466.90 GB. Extremes and historical/current billing ratios are explicitly exported for investigation. Current prices times tenure need not equal total historical charges because plans, discounts and prices may change. Without invoices, these records cannot be classified as errors, so no arbitrary capping or deletion is applied.')
    chart('monthly_charges',470,230)
    p('Domain checks cover age, tenure, nonnegative financial/behavioral quantities, satisfaction 1-10, discounts 0-100 and binary senior status. Observed category vocabularies are documented in the notebook; no authoritative external vocabulary was supplied.')
    title(3,'Observed churn patterns')
    chart('contract_type',470,210)
    p(dict(notes)['contract_type'])
    chart('satisfaction_score',470,210)
    p(dict(notes)['satisfaction_score'])
    p('The notebook also analyzes tenure, complaints, late payments, internet service, monthly usage, discounts, support, payment method, missingness and interactions. All exploratory outcome summaries use training customers only. Small subgroups and overlapping distributions limit interpretation.')
    title(4,'Features and model selection')
    p('A stratified 80/20 split locks 240 test customers before EDA. Five shuffled stratified folds on the 960 training customers evaluate all baseline and tuned candidates, with seed 42. Average precision selects the model; ROC-AUC, recall, F1 and top-20 precision qualify that decision.')
    p('Eight reusable features represent lifecycle bands, monthly charge per tenure, complaint and late-payment exposure over min(tenure,12), inverse satisfaction, a service/support gap, usage per charge, and a dissatisfaction/complaint interaction. Safe denominators and missingness handling are implemented in a serializable transformer. ID and target are excluded by an explicit feature allowlist.')
    table([['Model / version','CV AP mean (SD)','CV ROC','Top20 precision']]+[[r['model']+' / '+r['version'],f"{r['ap_mean']:.3f} ({r['ap_std']:.3f})",f"{r['roc_mean']:.3f}",f"{r['top20_mean']:.3f}"] for r in comparison.to_dict('records')],[165,135,85,105])
    p('Search spaces: Logistic C={0.1,1,10}, weights={none,balanced}; Forest depth={4,unlimited}, minimum leaf={2,8}, weights={none,balanced}, 120 trees; Boosting trees={60,120}, rate={0.03,0.1}, depth={1,2}. Every candidate uses five folds. Search results and training feature-ablation results are in the executed notebook.')
    p(f"Selected family: {s['winner']}; engineered features retained: {s['engineered']}. Features are selected by training CV. Fold dispersion supplies a robustness check. Winning CV estimates have search-selection optimism; they are distinct from the final held-out evaluation.")
    title(5,'Evaluation and probability reliability')
    table([['Metric','Test value']]+[[key,f'{m[key]:.4f}'] for key in ['accuracy','precision','recall','f1','roc_auc','average_precision','brier']],[260,230])
    chart('evaluation',490,170)
    p(f"The fixed classification threshold is {m['threshold']:.3f}, selected from training out-of-fold predictions by hypothetical net utility: $200 retained value, 25% success and $5 per contact. This unconstrained classifier is separate from the fixed 20% contact queue. False negatives lose potential retained value; false positives consume contact cost.")
    p('Calibration is checked using quantile bins and Brier score, alongside a constant-prevalence benchmark in the notebook. Few non-churn cases make reliability estimates imprecise. No post-test calibration or threshold changes were made. Accuracy alone is insufficient because predicting churn for everyone already yields approximately 86% accuracy.')
    p('Robustness: the CV comparison includes fold-to-fold standard deviations and a training-only feature ablation. One split, small minority class, missing timestamps and uncertain measurement timing prevent claims of temporal or external validity. Non-nested OOF threshold utility can retain model-selection optimism.')
    title(6,'Risk explanations and customer evidence')
    chart('importance',470,230)
    p('Global evidence is held-out raw-feature permutation AP loss, repeated eight times. Raw-column permutation aggregates encoded levels and propagates through derived features. Correlated predictors can share importance. These results are never used for feature selection.')
    for item in s['local']:
        p(f"{item['profile'].title()} risk: {item['customer_id']}, probability {item['probability']:.1%}. Strongest reference replacements: "+'; '.join(f'{name}: {delta:+.3f}' for name,delta in item['reference_replacement_deltas'][:3])+'.')
    p('Each delta is the original probability minus the score after replacing that raw attribute by its training median/mode, recalculating derived features. The notebook includes the actual records. This is a model-sensitivity explanation, not additive attribution or a causal treatment effect. Reference replacements can create implausible combinations. Medium denotes the cohort median score rank.')
    title(7,'Retention economics and operational next steps')
    p(f"Contact {int(t['contacted'])} customers, reaching {int(t['true_churners_reached'])} observed churners. Under the base assumptions, expected retained customers = {t['expected_retained']:.2f}; cost = ${t['cost']:.0f}; gross value = ${t['gross_value']:.0f}; net benefit = ${t['net_benefit']:.0f}. Incremental net value over uniform random outreach = ${t['incremental_net_vs_random']:.0f}.")
    table([['Success','Value','Model net','Random net','Difference']]+[[f"{r.success_rate:.0%}",f'${r.retained_value:.0f}',f'${r.model_net:.0f}',f'${r.random_net:.0f}',f'${r.incremental:.0f}'] for r in sensitivity.itertuples()],[75,75,110,110,120])
    for rec in s['recommendations']: p(rec)
    p('Assumptions are editable in CFG. These figures are hypothetical, not experimental ROI. Validate target provenance, scoring-time availability, temporal generalization, consent, fairness across age/region, offer costs and heterogeneous treatment effects. Run a randomized retention experiment; monitor calibration, data drift and customer burden before expansion.')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#197f86')); canvas.line(42,40,553,40); canvas.setFont('Helvetica',8); canvas.drawString(42,27,'RETENTION ANALYTICS | Observational evidence / hypothetical intervention'); canvas.drawRightString(553,27,str(doc.page))
    SimpleDocTemplate('04_business_report.pdf',pagesize=(595,842),rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=54).build(story,onFirstPage=footer,onLaterPages=footer)
