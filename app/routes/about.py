from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("about.html", {"request": request})
