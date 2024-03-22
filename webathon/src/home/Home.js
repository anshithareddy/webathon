import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHome, faInfoCircle, faUser, faCog } from '@fortawesome/free-solid-svg-icons';

const Sidebar = ({ features }) => {
  return (
    <aside className="sidebar">
      <ul className="sidebar-list">
        {features.map((feature) => (
          <li key={feature.id} className="sidebar-item">
            <FontAwesomeIcon icon={feature.icon} />
            {feature.title}
          </li>
        ))}
      </ul>
    </aside>
  );
};

export default Sidebar;