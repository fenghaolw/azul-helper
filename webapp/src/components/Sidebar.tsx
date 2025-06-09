interface SidebarProps {
  title: string;
  subtitle?: string;
  children: preact.ComponentChildren;
}

export function Sidebar({ title, subtitle, children }: SidebarProps) {
  return (
    <div className="sidebar">
      <div className="sidebar__header">
        <div className="sidebar__title-container">
          <div className="sidebar__accent"></div>
          <h2 className="sidebar__title">{title}</h2>
        </div>
        {subtitle && (
          <div className="sidebar__subtitle-container">
            <span className="sidebar__subtitle">{subtitle}</span>
          </div>
        )}
      </div>
      <div className="sidebar__content">{children}</div>
    </div>
  );
}
