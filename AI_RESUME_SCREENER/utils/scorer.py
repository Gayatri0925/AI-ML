def match_skills(resume_text, job_description):
    """
    Compare skills from resume with job description.
    Returns match percentage.
    """
    resume_words = set(resume_text.lower().split())
    job_words = set(job_description.lower().split())
    
    matched = resume_words & job_words
    score = len(matched) / len(job_words) * 100
    return round(score, 2)
