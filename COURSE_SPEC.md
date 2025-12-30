# CS 203: Software Tools and Techniques for AI
## Course Specification Document

**Institution:** IIT Gandhinagar
**Instructor:** Prof. Nipun Batra
**Version:** 1.0 (December 2025)

---

## 1. Audience Profile

### Primary Audience
- **First-year undergraduates** who have completed introductory Python programming
- Limited prior exposure to software engineering practices
- No assumed knowledge of HTTP, APIs, databases, or deployment

### Secondary Audience
- 2nd, 3rd, 4th year students with varying ML backgrounds
- May already know some tools; benefit from systematic coverage
- Can help mentor first-years in group projects

### Key Implications
- Never assume prerequisite knowledge beyond basic Python
- Provide 5-10 minute quick refreshers at the start of lectures when needed
- Build concepts from everyday analogies before formalizing
- Expect questions about "why do we need this?"

---

## 2. Pedagogical Philosophy

### Teaching Style: Intuition-First
Every technical concept should be introduced in this order:
1. **Everyday analogy** (restaurant, transportation, social media, gaming)
2. **The problem it solves** ("Why does this exist?")
3. **Visual mental model** (diagram showing how to think about it)
4. **Concrete example** (working code with minimal complexity)
5. **Formalization** (proper terminology, edge cases)

### Code Philosophy: Simple First
- Start with minimal working examples
- Mention production concerns in comments/notes
- Show "before/after" when teaching robustness
- Layered approach: add error handling, logging, etc. in later weeks

### Slide Density: Sparse, More Slides
- One main idea per slide
- Liberal use of slide breaks
- Build complex ideas across multiple slides progressively
- Easier for students to follow during lecture

### Historical Context
- Show evolution of tools (why we moved from X to Y)
- Help students understand the landscape
- "requests replaced urllib because..." type explanations

---

## 3. Running Examples

### Two-Track Approach

**Primary Track: Netflix Movie Predictor**
- Continuous project built week-by-week
- Each lecture adds a new capability
- Students see the full ML pipeline come together
- Familiar domain (everyone uses Netflix/streaming)

**Alternative Track: Rotating Domains**
Provide variety and cater to different interests:
- **E-commerce/Flipkart**: Product recommendations, reviews, pricing
- **Sports/IPL**: Cricket stats, player performance prediction
- **Music/Spotify**: Song recommendations, playlist analysis
- **News/Social**: Sentiment analysis, trending topics

Each week's lab should offer at least 2 domain options.

---

## 4. Content Structure

### Lecture Format
Each lecture should include:

1. **Title Slide** (styled with `title-slide` class)
2. **Quick Refresher** (5-10 min if prerequisites needed)
3. **Motivation** ("Imagine you're at Netflix...")
4. **Intuition Section** (analogies, visual mental models)
5. **Core Content** (concepts + code examples)
6. **Common Errors** (what breaks and why, side-by-side format)
7. **Brief Tool Comparison** (why we chose X over Y)
8. **Interview Question** (1-2 questions relevant to this topic)
9. **Key Takeaways** (essential, every lecture)
10. **Lab Preview** (what they'll build)

### Slide Elements

**Callout Boxes:**
- `<div class="insight">` - Key intuitions, mental models
- `<div class="example">` - Concrete examples
- `<div class="warning">` - Common mistakes, gotchas
- `<div class="realworld">` - Industry context, case studies

**Code Blocks:**
- VS Code Dark+ theme (already configured in CSS)
- Minimal comments (code should be self-explanatory)
- Maximum ~15 lines per slide
- Split longer code across multiple slides

**Diagrams:**
- Prefer ASCII diagrams for portability and height control
- Professional PNG/SVG for complex system architectures
- Real terminal screenshots for error messages (annotated)
- Side-by-side format for error explanations

---

## 5. Error Handling & Debugging

### Show Common Errors
Every major topic should include:
1. What the error looks like (real terminal output)
2. Why it happens (root cause)
3. How to fix it (solution)

### Side-by-Side Format
```
| Error                    | Fix                        |
|--------------------------|----------------------------|
| `KeyError: 'director'`   | Use `.get('director', '')` |
| `ConnectionTimeout`      | Add `timeout=10` parameter |
```

### Debug Mindset
Cultivate systematic debugging:
- Read the error message (what line? what type?)
- Isolate the problem (print statements, breakpoints)
- Search effectively (how to Google error messages)
- Ask for help properly (include code, error, what you tried)

---

## 6. Tool Selection Philosophy

### Brief Comparisons
When introducing a tool, briefly mention alternatives:
- "We use `requests` because it's simpler than `urllib`"
- "FastAPI is newer and has better docs than Flask"
- "pandas is the standard for tabular data in Python"

### Core Tools (Deep Mastery)
Focus on mastering these thoroughly:
1. **requests** - HTTP client
2. **pandas** - Data manipulation
3. **BeautifulSoup** - Web scraping
4. **FastAPI** - API building
5. **pytest** - Testing
6. **Git** - Version control (learn by doing)
7. **Docker** - Containerization (introduced, not deep)

### Breadth vs Depth: Deep on Core
- Master 3-4 key tools per topic area
- Mention alternatives for awareness
- Don't try to teach everything

---

## 7. Labs & Assignments

### Lab Design: Scaffolded Projects
- Jupyter notebooks with guided structure
- Clear TODO sections for students to fill in
- Starter code provided
- Expected output shown
- Hints available (progressive disclosure)

### Group Projects
- Teams of 3-4 students
- Encourage peer learning
- Mixed experience levels when possible
- Clear role assignments

### Workload: 3-4 Hours/Week
- Achievable for first-years
- Room for deeper exploration for advanced students
- Weekly assignments with immediate feedback

### Computing Resources
Students have access to:
- Personal laptops (varying specs)
- Lab computers (standardized environment)
- Cloud credits (Google Colab)

Provide detailed setup guides for all three environments.

---

## 8. Assessment

### Weekly Assignments
- Regular practice reinforces learning
- Immediate feedback loop
- Builds toward larger projects
- Auto-grading where possible

### Assessment Components (Suggested)
- Weekly labs: 40%
- Mid-term project: 20%
- Final project: 30%
- Participation/peer review: 10%

---

## 9. Addressing Pain Points

### Setup Struggles
- Detailed step-by-step guides for Windows/Mac/Linux
- requirements.txt emphasis from week 1
- Troubleshooting FAQs
- Office hours focused on setup in first weeks

### Concept Gaps
- Quick refreshers at lecture start
- Link to prerequisite resources
- Integrated teaching when needed
- Don't assume HTTP, JSON, etc. knowledge

### Overwhelm
- One tool at a time
- Build complexity gradually
- Clear learning objectives per lecture
- "You don't need to memorize this" reminders

### Theory-Practice Disconnect
- Labs directly practice lecture content
- Scaffolded notebooks bridge the gap
- Live coding demonstrations
- "Type along with me" sections

---

## 10. Tone & Style

### Formal Academic
- Professional language
- Proper technical terminology
- Rigorous but accessible
- Avoid excessive casualness

### Include When Natural
- Everyday analogies (food, transport, social media, gaming)
- Brief humor where appropriate
- Encouraging acknowledgment of challenges
- "This is tricky, but you'll get it"

---

## 11. Interview Questions

Each lecture should include 1-2 relevant interview questions:

**Example for Week 1 (Data Collection):**
> "Explain the difference between GET and POST requests."
> "How would you handle rate limiting when collecting data from an API?"

**Example for Week 10 (HTTP APIs):**
> "What HTTP status code would you return if a user sends invalid JSON?"
> "Explain what makes an API 'RESTful'."

---

## 12. Key Takeaways Format

Every lecture ends with a Key Takeaways slide:

```markdown
# Key Takeaways

1. **First major concept** - One sentence summary
2. **Second major concept** - One sentence summary
3. **Third major concept** - One sentence summary
4. **Practical skill** - "You can now do X"

**Next week:** Brief preview of upcoming topic
```

---

## 13. Ethics Integration

### Light Touch Approach
- Mention ethical considerations where directly relevant
- Don't dedicate entire lectures to ethics
- Integrate naturally into discussions

**Examples:**
- Data collection: Respect robots.txt, ToS, rate limits
- Web scraping: Legal considerations, don't harm servers
- APIs: Handle user data responsibly
- Deployment: Consider accessibility, bias in models

---

## 14. Learning Outcomes

By course end, students should feel:

1. **End-to-End Capability**
   "I can build a complete ML system from data collection to deployment"

2. **Tool Confidence**
   "I know which tool to use for any data/ML task"

3. **Debug Mindset**
   "I can figure out why things break and fix them systematically"

4. **Professional Practices**
   "I write code that others can understand, maintain, and reproduce"

---

## 15. Week-by-Week Overview

| Week | Topic | Primary Example | Key Tools |
|------|-------|-----------------|-----------|
| 1 | Data Collection | Netflix movies from OMDb | requests, curl, DevTools |
| 2 | Data Validation | Cleaning movie data | Pydantic, pandas |
| 3 | Data Labeling | Rating movies | Label Studio basics |
| 4 | Optimizing Labeling | Active learning | modAL, sampling |
| 5 | Data Augmentation | Expanding dataset | nlpaug, albumentations |
| 6 | LLM APIs | GPT for features | OpenAI API, prompting |
| 7 | Model Development | Training predictor | scikit-learn, MLflow |
| 8 | Reproducibility | Reproducible pipeline | DVC, seeds, configs |
| 9 | Interactive Demos | Streamlit app | Streamlit, Gradio |
| 10 | HTTP APIs | FastAPI service | FastAPI, Pydantic |
| 11 | Git & CI/CD | Automated testing | Git, GitHub Actions |
| 12 | Deployment | Cloud deployment | Docker, cloud basics |
| 13 | Profiling & Optimization | Making it fast | cProfile, optimization |
| 14 | Model Monitoring | Production monitoring | logging, metrics |
| 15 | Course Summary | Complete pipeline | Integration |

---

## 16. Visual Design Guidelines

### ASCII Diagrams
Preferred for flow and concept diagrams:
```
Client ──request──▶ API ──query──▶ Database
       ◀──response──    ◀──data───
```

### Professional Diagrams
Use for:
- System architectures
- Complex data flows
- Comparison matrices

### Screenshots
Use for:
- Error messages (annotated)
- Tool interfaces (DevTools, IDEs)
- Expected output

### Tables
Use for:
- Comparisons (API vs Scraping)
- Reference information (HTTP status codes)
- Feature matrices

---

## 17. Reproducibility Requirements

### Every Code Example Should:
- Work with specified versions in requirements.txt
- Use seed values where randomness is involved
- Have clear input/output expectations
- Be copy-paste runnable

### Environment Setup:
- requirements.txt with pinned versions
- Clear Python version requirement
- OS-specific notes where needed

---

## 18. Continuous Improvement

### Feedback Collection
- End-of-lecture quick polls
- Weekly anonymous feedback form
- Mid-semester course corrections
- End-of-semester comprehensive review

### Iteration Plan
- Update examples with current tools annually
- Refresh real-world case studies
- Incorporate student suggestions
- Track which concepts need more coverage

---

## Appendix A: Analogy Bank

### Food/Restaurant
- API = Waiter (takes order, brings food, you don't see kitchen)
- Rate limiting = Restaurant seating capacity
- Authentication = Reservation system
- Caching = Keeping popular dishes ready

### Transportation
- Data pipeline = Assembly line
- API endpoint = Bus stop
- Load balancing = Traffic routing
- Containers = Shipping containers

### Social Media
- API rate limits = Post frequency limits
- Webhooks = Notifications
- Caching = Feed pre-loading
- Authentication = Login sessions

### Gaming
- Checkpoints = Model checkpoints
- Levels = Progressive complexity
- Multiplayer = Distributed systems
- Save states = Reproducibility

---

## Appendix B: Common Errors Reference

### HTTP Errors
| Code | Meaning | Common Cause |
|------|---------|--------------|
| 400 | Bad Request | Malformed JSON |
| 401 | Unauthorized | Missing API key |
| 403 | Forbidden | Invalid permissions |
| 404 | Not Found | Wrong endpoint |
| 429 | Rate Limited | Too many requests |
| 500 | Server Error | Their problem |

### Python Errors
| Error | Common Cause | Fix |
|-------|--------------|-----|
| KeyError | Missing dict key | Use .get() |
| JSONDecodeError | Invalid JSON | Check response.text first |
| ConnectionError | Network issue | Add retry logic |
| Timeout | Slow server | Increase timeout |

---

*This specification guides the creation and maintenance of CS 203 course materials. Review and update annually.*
