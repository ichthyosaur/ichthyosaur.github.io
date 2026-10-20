# Loaded explicitly by integration tests, never by normal site builds.
# Generate fixtures in memory so tests do not depend on published demo posts.
require 'jekyll'

module IntegrationFixtures
  class Pages < Jekyll::Generator
    def generate(site)
      fixtures = {
        'giscus' => { 'giscus_comments' => true },
        'disqus' => { 'disqus_comments' => true },
        'distill' => { 'layout' => 'distill', 'giscus_comments' => true, 'mermaid' => { 'enabled' => true }, 'tikzjax' => true },
        'rtl' => { 'lang' => 'fa' },
        'marimo' => { 'marimo' => true }
      }
      fixtures.each do |name, data|
        page = Jekyll::PageWithoutAFile.new(site, site.source, "integration-fixtures/#{name}", 'index.html')
        page.data = { 'layout' => 'post', 'title' => "Integration fixture: #{name}", 'date' => Time.utc(2020, 1, 1) }.merge(data)
        page.content = '<p>Integration test fixture.</p>'
        site.pages << page
      end
    end
  end
end
