using Microsoft.AspNetCore.Mvc;

namespace LibrasAI.Controllers;

public sealed class HomeController : Controller
{
    public IActionResult Index() => View();

    public IActionResult Error() => View();
}
