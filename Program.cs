using System;
using Furb.Pos.DataScience.PCA;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;

namespace Furb.Pos.DataScience
{

   class Program
    {

        static void Main(string[] args)
        {
            var serviceCollection = new ServiceCollection();

            ConfigureServices(serviceCollection);

            var serviceProvider = serviceCollection.BuildServiceProvider();

            var pca = serviceProvider.GetService<PCABoostrap>();

            pca.Run(7);
        }

        private static void ConfigureServices(ServiceCollection services)
        {
            services.AddLogging(configure => configure.AddConsole())
                .Configure<LoggerFilterOptions>(options => options.MinLevel = LogLevel.Information)
                .AddTransient<PCABoostrap>()
                .AddTransient<PCAEigenFace>();
        }

    }
}
