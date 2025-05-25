# Setting Up Your Marker Document Extractor on GitHub

This guide will help you upload your Marker Document Extractor project to a new GitHub repository.

## 🚀 Quick Setup

### Step 1: Create a New GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Choose a repository name (e.g., `marker-document-extractor`)
5. Add a description: "Web-based document extraction tool using Marker library"
6. Choose visibility (Public or Private)
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

### Step 2: Initialize Git and Push to GitHub

Open terminal in your project directory and run:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Marker Document Extractor web application"

# Add your GitHub repository as remote (replace with your actual repository URL)
git remote add origin https://github.com/yourusername/marker-document-extractor.git

# Push to GitHub
git push -u origin main
```

### Step 3: Set Up Environment Variables

**Important**: Your `.env` file with API keys is excluded from git for security. Users will need to:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add their API keys:
   ```env
   GOOGLE_API_KEY=their_actual_api_key_here
   ```

## 📝 Repository Structure

After upload, your repository will contain:

```
marker-document-extractor/
├── README.md              # Main documentation
├── SETUP_GITHUB.md       # This setup guide
├── LICENSE               # MIT License
├── .gitignore            # Git ignore rules
├── .env.example          # Environment template
├── requirements.txt      # Python dependencies
├── main.py              # FastAPI application
├── index.html           # Web interface
└── user_preferences.json # User preferences (excluded from git)
```

## 🔒 Security Best Practices

### Files Excluded from Git:
- `.env` - Contains sensitive API keys
- `user_preferences.json` - User-specific settings
- `__pycache__/` - Python bytecode
- `extracted_images/` - Temporary image files
- `.vscode/` - IDE settings

### What to Include in Documentation:
- ✅ Installation instructions
- ✅ Usage examples
- ✅ Configuration templates
- ❌ Actual API keys
- ❌ Personal preferences

## 🌟 Repository Enhancements

Consider adding these features to make your repository more professional:

### GitHub Actions (CI/CD)
Create `.github/workflows/ci.yml`:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python -c "import fastapi; print('Dependencies OK')"
```

### Issue Templates
Create `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md`

### Contributing Guidelines
Create `CONTRIBUTING.md` with contribution instructions

### GitHub Pages Demo
Set up GitHub Pages to host a demo version (without API functionality)

## 📊 Repository Settings

### Recommended Settings:
1. **Issues**: Enable for bug tracking
2. **Projects**: Enable for task management
3. **Wiki**: Enable for extended documentation
4. **Discussions**: Enable for community Q&A
5. **Security**: Enable vulnerability alerts

### Branch Protection:
1. Go to Settings → Branches
2. Add rule for `main` branch
3. Require pull request reviews
4. Require status checks to pass

## 🎯 Next Steps

After setting up the repository:

1. **Add Topics**: Go to repository settings and add relevant topics:
   - `document-extraction`
   - `pdf-processing`
   - `marker`
   - `fastapi`
   - `web-application`

2. **Create Releases**: Tag stable versions:
   ```bash
   git tag -a v1.0.0 -m "Initial release"
   git push origin v1.0.0
   ```

3. **Monitor Issues**: Respond to user questions and bug reports

4. **Update Documentation**: Keep README and docs up to date

## 🆘 Troubleshooting

### Common Git Issues:

**Repository already exists error**:
```bash
git remote rm origin
git remote add origin https://github.com/yourusername/new-repo-name.git
```

**Large file errors**:
```bash
git rm --cached large-file.pdf
git commit -m "Remove large file"
```

**Authentication issues**:
- Use personal access tokens instead of passwords
- Set up SSH keys for easier authentication

### Pre-upload Checklist:
- [ ] Remove sensitive data from all files
- [ ] Test that the application works locally
- [ ] Verify all dependencies are in requirements.txt
- [ ] Check that .gitignore covers sensitive files
- [ ] Update README with accurate installation steps
- [ ] Add appropriate license file

---

Your Marker Document Extractor is now ready for GitHub! 🎉
