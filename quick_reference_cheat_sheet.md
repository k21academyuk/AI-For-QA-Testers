# AI Tester Interview - Quick Reference Cheat Sheet

## 🎯 Key Concepts at a Glance

### LLM Testing Fundamentals
```
Temperature:
├── 0.0   → Deterministic (same input = same output)
├── 0.7   → Balanced (default for most uses)
└── 2.0   → Very creative (high randomness)

Critical Test Areas:
✓ Consistency (temp=0 validation)
✓ Safety (no harmful content, proper disclaimers)
✓ Accuracy (fact-checking against references)
✓ Completeness (answers full query)
✓ Bias detection (demographic fairness)
```

### API Testing Essentials
```
HTTP Status Codes:
200 → Success
400 → Bad Request (client error)
401 → Unauthorized
429 → Rate Limited
500 → Server Error

Key Validations:
✓ Status code
✓ Response schema
✓ Response time
✓ Error messages
✓ Authentication
```

### Test Automation Hierarchy
```
Performance Tests (1%)     ← Slowest, few tests
    ↑
Integration Tests (20%)
    ↑
Unit/API Tests (70%)       ← Fastest, most tests
    ↑
Static Analysis (100%)     ← Always run
```

### CI/CD Pipeline Stages
```
1. Commit → 2. Build → 3. Test → 4. Deploy → 5. Monitor
            
Test Stages:
- Smoke tests (5 min) → Critical path only
- Regression (30 min) → Full suite
- Performance (1 hr) → Load testing
```

---

## 💡 Common Interview Questions - Quick Answers

### "How do you test non-deterministic systems?"
**3-Point Answer:**
1. Control temperature=0 for deterministic testing
2. Statistical analysis for higher temperatures (variance, distribution)
3. Semantic similarity checks vs exact matching

### "Explain your automation framework approach"
**Framework Pillars:**
- **Structure**: Page Object Model for APIs
- **Data**: Parameterized tests, external test data
- **Reporting**: Allure/HTML reports integrated in CI/CD
- **Maintenance**: DRY principles, reusable fixtures

### "How do you prioritize testing?"
**Risk-Based Matrix:**
```
                High Impact    Low Impact
High Probability   P0 - NOW      P1 - Next
Low Probability    P2 - Later    P3 - Maybe
```
**Factors**: Safety, compliance, frequency, detection difficulty

### "Describe your API testing strategy"
**5-Layer Approach:**
1. **Contract**: Schema validation
2. **Functional**: Feature correctness
3. **Security**: Auth, injection attempts
4. **Performance**: Response times, load
5. **Reliability**: Error handling, retries

---

## 🔧 Python Testing Snippets

### Basic API Test Template
```python
import pytest
import requests

@pytest.fixture
def api_client():
    return {"url": BASE_URL, "key": API_KEY}

def test_api_call(api_client):
    response = requests.post(
        f"{api_client['url']}/completions",
        headers={"Authorization": f"Bearer {api_client['key']}"},
        json={"prompt": "test", "temperature": 0.0}
    )
    assert response.status_code == 200
    assert "choices" in response.json()
```

### Response Validation
```python
def validate_medical_response(response_text):
    checks = {
        "has_disclaimer": any(
            phrase in response_text.lower() 
            for phrase in ["consult doctor", "healthcare provider"]
        ),
        "appropriate_length": 20 < len(response_text.split()) < 500,
        "no_diagnosis": "you have" not in response_text.lower()
    }
    return all(checks.values()), checks
```

### Retry Logic
```python
from functools import wraps
import time

def retry_on_rate_limit(max_retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitError:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise
        return wrapper
    return decorator
```

---

## 📊 Pharmaceutical Testing Checklist

### Regulatory Compliance Tests
- [ ] HIPAA: No PII in responses
- [ ] FDA 21 CFR Part 11: Audit trails
- [ ] GxP: Validation documentation
- [ ] Medical disclaimers on all health advice

### Safety-Critical Validations
- [ ] No definitive diagnoses
- [ ] Appropriate referrals to healthcare providers
- [ ] Drug interaction warnings present
- [ ] Contraindication information accurate
- [ ] Dosage recommendations include disclaimers

### Accuracy Benchmarks
- [ ] Cross-reference with FDA drug labels
- [ ] Validate against medical textbooks
- [ ] Expert review of sample responses
- [ ] Semantic similarity to authoritative sources >85%

---

## 🎤 STAR Method Examples

### Situation-Task-Action-Result Framework

**Example 1: Finding Critical Bug**
- **S**: LLM provided dosages without warnings
- **T**: Assess scope, prevent harm, implement fix
- **A**: Documented, analyzed logs (2% affected), created tests, collaborated on fix, added monitoring
- **R**: Fixed in 48hrs, 50+ new safety tests, no recurrence

**Example 2: Disagreement on Bug Severity**
- **S**: Developer marked disclaimer omission as "Low"
- **T**: Advocate for proper severity in pharma context
- **A**: Gathered evidence, showed impact data, consulted compliance, presented case
- **R**: Bug upgraded to Critical, fixed quickly, clearer guidelines established

**Example 3: Time-Constrained Testing**
- **S**: Major release in 2 days
- **T**: Validate critical functionality
- **A**: Risk-based prioritization, automated regression, focused on safety/compliance, documented coverage gaps
- **R**: Released on time, zero critical issues, documented tech debt

---

## 🔍 Common Pitfalls to Avoid

### In Testing
❌ Testing only happy paths
❌ Ignoring edge cases (empty strings, special chars)
❌ Not validating error messages
❌ Forgetting performance/load tests
❌ No negative testing

✅ Comprehensive test scenarios
✅ Boundary value analysis
✅ Clear, actionable error validation
✅ Performance benchmarks from start
✅ Security & abuse testing

### In Interviews
❌ "I know everything"
❌ Criticizing previous employers
❌ Vague answers without examples
❌ Not asking questions
❌ Being defensive about gaps

✅ "I'm continuously learning"
✅ "I learned from that experience"
✅ Specific STAR stories
✅ Thoughtful questions prepared
✅ Honest about limitations, show growth mindset

---

## 📋 Questions to Ask Interviewer

### Technical Environment
1. "What LLM models/APIs are you currently using?"
2. "What's your current test automation coverage?"
3. "How do you handle model version updates?"
4. "What's your CI/CD pipeline setup?"

### Team & Process
5. "How does QA collaborate with ML engineers?"
6. "What's your deployment frequency?"
7. "How do you balance speed with thorough testing?"
8. "What's the team's approach to test automation vs manual?"

### Role-Specific
9. "What are the biggest testing challenges you're facing?"
10. "How is success measured for this role in the first 90 days?"
11. "What compliance/regulatory frameworks do you follow?"
12. "What opportunities for growth exist in this role?"

---

## 🚀 Day-Of-Interview Checklist

### 2 Hours Before
- [ ] Review this cheat sheet
- [ ] Review job description
- [ ] Test video/audio setup
- [ ] Prepare notepad and pen
- [ ] Have water nearby

### 30 Minutes Before
- [ ] Review your resume
- [ ] Read latest company news
- [ ] Practice breathing/relaxation
- [ ] Close unnecessary apps
- [ ] Join 5 minutes early

### During Interview
- [ ] Take notes on questions
- [ ] Ask for clarification if needed
- [ ] Use STAR method for behavioral Qs
- [ ] Show enthusiasm
- [ ] Ask your prepared questions

### After Interview
- [ ] Send thank-you email within 24hrs
- [ ] Reflect on what went well
- [ ] Note areas for improvement
- [ ] Follow up per their timeline

---

## 💪 Confidence Boosters

**Remember:**
- You've prepared thoroughly
- Your experience is valuable
- It's okay to say "I don't know, but here's how I'd find out"
- They're evaluating fit, not perfection
- You're also interviewing them

**Your Value Proposition:**
- Automation expertise
- API testing skills
- Understanding of AI/LLM behavior
- Quality-first mindset
- Pharmaceutical awareness

**If Nervous:**
1. Take deep breath
2. Remember: conversation, not interrogation
3. They want you to succeed
4. Focus on what you DO know
5. Enthusiasm > perfection

---

## 📚 Last-Minute Review Topics

### Must Know Cold
- Pytest basics and fixtures
- API testing fundamentals
- CI/CD pipeline stages
- LLM temperature parameter
- Medical disclaimer requirements

### Should Be Comfortable With
- Automation framework design
- Response validation techniques
- Model comparison testing
- Risk-based test prioritization
- Pharmaceutical compliance basics

### Nice to Have
- Advanced prompt engineering
- Statistical validation methods
- Performance testing tools
- Bias detection techniques
- Specific pharma regulations

---

## 🎯 Final Reminders

1. **Be Specific**: Use concrete examples from experience
2. **Show Process**: Explain your thinking, not just results
3. **Ask Questions**: Shows engagement and critical thinking
4. **Stay Positive**: Even when discussing challenges
5. **Be Yourself**: Authenticity > rehearsed perfection

**You've got this! 🌟**

Good luck!
